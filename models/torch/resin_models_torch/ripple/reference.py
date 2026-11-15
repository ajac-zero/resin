"""Readable tokenizer-free Ripple loading and execution on PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from resin import CacheLayout, RunnerConfig
from resin.backend.loading import (
    assign_model_tensor,
    load_safetensor_weights,
    read_config,
    resolve_dtype,
)
from resin.backend.plans import BatchView
from resin.backend.sampling import (
    PromptScores,
    TorchSampler,
    format_output,
    prompt_scores,
)
from resin.backend.topology import ParallelismCapabilities
from resin.models.ripple.plan import RippleBatchPlan, build_ripple_batch_plan
from resin.protocols import Runner
from resin_models_torch.ripple.config import RippleConfig
from resin_models_torch.ripple.model import RippleForCausalLM, RippleState

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput


def _load_prefix(model_path: Path, model: torch.nn.Module, device, dtype) -> None:
    """Read only the tensors ``model`` owns, opening only the shards that hold them."""

    from resin.backend.safetensors import SafetensorReader

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        load_safetensor_weights(
            model_path,
            model,
            device=device,
            dtype=dtype,
            checkpoint_keys=list(model.state_dict()),
        )
        return
    weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    wanted = list(model.state_dict())
    if missing := [name for name in wanted if name not in weight_map]:
        raise ValueError(f"checkpoint is missing tensors: {missing[:8]}")
    by_file: dict[str, list[str]] = {}
    for name in wanted:
        by_file.setdefault(weight_map[name], []).append(name)
    with SafetensorReader() as reader:
        for filename, names in by_file.items():
            path = model_path / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint shard: {path}")
            for name in names:
                assign_model_tensor(
                    model, name, reader.read(path, name), device=device, dtype=dtype
                )


def load_model(
    model_path: Path,
    *,
    device,
    dtype=None,
    num_layers: int | None = None,
) -> tuple[RippleForCausalLM, RippleConfig]:
    """Load a Ripple checkpoint, or its first ``num_layers`` layers.

    A reduced load keeps the embeddings, final norm and LM head and never opens
    shards that only hold later layers.
    """

    raw = read_config(model_path)
    if raw.get("quantization") or raw.get("quantization_config"):
        raise ValueError("the PyTorch runner requires an unquantized Ripple checkpoint")
    config = RippleConfig.from_dict(raw)
    if num_layers is not None:
        config = config.same_shape_reduced(num_layers)
    target = torch.device(device)
    load_target = target if target.type in {"cpu", "cuda"} else torch.device("cpu")
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None and not resolved_dtype.is_floating_point:
        raise ValueError("model dtype must be floating point")
    with torch.device("meta"):
        model = RippleForCausalLM(config)
    if num_layers is None:
        load_safetensor_weights(
            model_path, model, device=load_target, dtype=resolved_dtype
        )
    else:
        _load_prefix(model_path, model, load_target, resolved_dtype)
    if target != load_target:
        model.to(target)
    model.eval()
    return model, config


class RippleTorchRunner(Runner):
    """Ripple runner consuming Rust-built integer-token batches.

    With a ``cache_layout`` the Engine leases real full-attention and window
    block tables; each plan is translated and validated, while the KV values
    themselves stay in request-owned PyTorch state.
    """

    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model: str | Path | RippleForCausalLM,
        *,
        device: str | torch.device = "cpu",
        dtype=None,
        num_layers: int | None = None,
        cache_layout: CacheLayout | None = None,
        prefill_chunk_size: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        if isinstance(model, RippleForCausalLM):
            self._model, self.config = model, model.config
        else:
            self._model, self.config = load_model(
                Path(model), device=self.device, dtype=dtype, num_layers=num_layers
            )
        self.cache_layout = cache_layout
        self.runner_config = RunnerConfig(
            continuation_prefill=True,
            cache_layout=cache_layout,
            prefill_chunk_size=prefill_chunk_size,
        )
        self.eos_token_id = list(self.config.eos_token_ids)
        self.vocab_size = self.config.vocab_size
        self._sampler = TorchSampler()
        self._cache: dict[int, RippleState] = {}
        self.forward_calls = 0
        self.plans: list[RippleBatchPlan] = []

    @staticmethod
    def bind_resources(_binding) -> None:
        """KV lives in request-owned tensors; the layout only drives scheduling."""

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self.forward_calls += 1
        batch = BatchView.from_plan(plan)
        if self.cache_layout is not None:
            self.plans.append(build_ripple_batch_plan(plan, self.cache_layout))
        for index, sequence_id in enumerate(batch.sequence_ids):
            sequence_id = int(sequence_id)
            cache = self._cache.get(sequence_id)
            offset = 0 if cache is None else cache.length
            token_slice = batch.token_slice(index)
            positions = batch.positions[token_slice]
            if int(batch.context_lengths[index]) != offset:
                expected = np.arange(offset, offset + len(positions), dtype=np.uint32)
                raise ValueError(
                    f"non-contiguous positions for request {sequence_id}: "
                    f"expected {expected.tolist()}, got {positions.tolist()}"
                )
            if np.any(batch.token_ids[token_slice] >= self.vocab_size):
                raise ValueError(f"token id is out of range for request {sequence_id}")

        logits_rows: list[torch.Tensor] = []
        prompt_rows = []
        with torch.inference_mode():
            if batch.num_sequences > 1 and np.all(batch.query_lengths == 1):
                caches = [self._state(int(s)) for s in batch.sequence_ids]
                inputs = torch.tensor(
                    batch.token_ids[:, None], dtype=torch.long, device=self.device
                )
                positions = torch.tensor(
                    batch.positions[:, None], dtype=torch.long, device=self.device
                )
                logits = self._model.forward_batch(inputs, caches, positions)
                logits_rows.append(logits[:, -1, :])
                if plan.needs_prompt_logprobs:
                    prompt_rows.append(prompt_scores(plan, logits[:, 0]))
            else:
                self._forward_rows(batch, logits_rows, plan, prompt_rows)
        logits = torch.cat(logits_rows)
        scores = PromptScores.concat(cast(list[PromptScores], prompt_rows))
        return format_output(
            logits, self._sampler(logits, plan), plan, prompt_logprobs=scores
        )

    def _state(self, sequence_id: int) -> RippleState:
        state = self._cache.get(sequence_id)
        if state is None:
            state = self._cache[sequence_id] = self._model.make_cache()
        return state

    def _forward_rows(self, batch, logits_rows, plan, prompt_rows) -> None:
        for index, sequence_id in enumerate(batch.sequence_ids):
            token_slice = batch.token_slice(index)
            inputs = torch.tensor(
                batch.token_ids[token_slice][None], dtype=torch.long, device=self.device
            )
            positions = torch.tensor(
                batch.positions[token_slice][None], dtype=torch.long, device=self.device
            )
            logits = self._model(
                inputs,
                self._state(int(sequence_id)),
                positions,
                logits_to_keep=None if plan.needs_prompt_logprobs else 1,
                return_hidden=plan.needs_prompt_logprobs,
            )
            if plan.needs_prompt_logprobs:
                prompt_rows.append(
                    prompt_scores(
                        plan,
                        logits[0],
                        self._model.logits,
                        token_start=token_slice.start,
                    )
                )
                logits = self._model.logits(logits[:, -1:])
            logits_rows.append(logits[:, -1, :])

    def release(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            self._cache.pop(request_id, None)

    @property
    def live_cache_count(self) -> int:
        return len(self._cache)
