"""Ripple BF16 CUDA runner for Ampere (SM80) and newer GPUs.

Attention uses FlashInfer paged BF16 attention with a standard
``1 / sqrt(head_dim)`` softmax scale (unlike Gemma 4, which pre-scales Q).
Nothing here needs Blackwell, cuTile, FA4, Quack or NVFP4: the norms are plain
torch ops matching the reference rounding, the sampler's greedy path is Triton,
and FlashInfer is imported lazily when the runner is constructed.

Scoring note: ``format_output`` only invokes the CuTeDSL scoring kernels (which
need ``nvidia-cutlass-dsl`` and ``quack-kernels``; the kernels are tested on
SM80+) when a request asks for logprobs, top logprobs or prompt logprobs.
Greedy and temperature sampling without logprobs never touch CuTeDSL.
Top-p sampling uses the CuTeDSL nucleus kernel (also SM80+). Raw logits can be
inspected without scoring through ``RippleCudaRunner.logits_observer``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from resin import BatchPhase, RunnerConfig
from resin.backend.cache_storage import byte_regions, live_rows
from resin.backend.capacity import CacheCapacityPlan, plan_cache_capacity
from resin.backend.cuda.failures import GraphInputError
from resin.backend.cuda.graphs import (
    DEFAULT_MAX_NUM_SEQS,
    CudaGraphMode,
    CudaGraphStep,
    GraphShape,
    resolve_decode_graph_shapes,
)
from resin.backend.cuda.metrics import observe_execution
from resin.backend.cuda.parallel import (
    TensorParallelContext,
    TensorParallelLayout,
    replicated_cache_capacity,
)
from resin.backend.cuda.runner import (
    CudaRunner,
    capture_sampling_inputs,
    pad_sampling_inputs,
)
from resin.backend.cuda.scoring import format_output, prompt_scores
from resin.backend.resources import CachePoolBinding
from resin.backend.sampling import SAMPLING_OFFSET_STRIDE, token_output
from resin.backend.topology import ParallelGroups, ParallelismCapabilities
from resin.models.gemma4.plan import PagedAttentionPlan
from resin.models.ripple.config import (
    FULL_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    RippleConfig,
    attention_geometry,
)
from resin.models.ripple.plan import (
    RippleBatchPlan,
    build_ripple_batch_plan,
    ripple_cache_layout,
)
from resin_models_cuda.ripple.checkpoint import (
    DEFAULT_MODEL,
    load_ripple_cuda_model,
    resolve_model_path,
)
from resin_models_cuda.ripple.model import RippleCudaForCausalLM

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput

PAGE_SIZE = 64
PREFILL_CHUNK_SIZE = 8192
_GRAPH_WORKSPACE_BYTES = 128 * 1024 * 1024
_PREFILL_WORKSPACE_BYTES_PER_ELEMENT = 16


def _prefill_workspace_bytes(chunk_size: int, query_heads: int, head_dim: int) -> int:
    return chunk_size * query_heads * head_dim * _PREFILL_WORKSPACE_BYTES_PER_ELEMENT


def plan_ripple_kv_capacity(
    total_memory_bytes: int,
    free_memory_bytes: int,
    static_memory_fraction: float,
    *,
    config: RippleConfig,
    page_size: int = PAGE_SIZE,
    prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
    workspace_bytes: int | None = None,
    cache_element_bytes: int = 2,
    tensor_parallel_size: int = 1,
) -> CacheCapacityPlan:
    """Size the parent pool: one parent holds every sliding layer's page.

    Every layer stores the same KV width, so a parent is ``local_layer_count``
    layers of one window page, or ``history_blocks_per_parent`` full-attention
    history pages (4 on Ripple-65B, whose period is ``[S, S, S, S, G]``).
    """

    if type(page_size) is not int or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if cache_element_bytes not in {1, 2}:
        raise ValueError("cache_element_bytes must be one or two")
    geometry = attention_geometry(config, TensorParallelLayout(0, tensor_parallel_size))
    bytes_per_parent = (
        2
        * config.local_layer_count
        * page_size
        * geometry.kv_heads
        * config.head_dim
        * cache_element_bytes
    )
    if workspace_bytes is None:
        workspace_bytes = 2 * _GRAPH_WORKSPACE_BYTES + 2 * _prefill_workspace_bytes(
            prefill_chunk_size, geometry.q_heads, config.head_dim
        )
    return plan_cache_capacity(
        total_memory_bytes,
        free_memory_bytes,
        static_memory_fraction,
        bytes_per_parent,
        overhead_bytes=workspace_bytes,
    )


class _RippleKvArena:
    """One fungible physical slab shared by the full and window cache groups.

    Window pages view a parent as ``local_layer_count`` layers of one page;
    full-attention pages view it as ``history_blocks_per_parent`` pages of
    ``global_layer_count`` layers. Both are ``(layers, pages, page_size,
    kv_heads, head_dim)`` in NHD layout.
    """

    def __init__(
        self,
        device,
        config: RippleConfig,
        parent_capacity: int,
        *,
        kv_heads: int,
        page_size: int = PAGE_SIZE,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.dtype = dtype
        self.page_size = page_size
        self.parent_capacity = parent_capacity
        physical_parent_capacity = parent_capacity + 1  # parent zero is graph padding
        self.storage = torch.empty(
            (
                physical_parent_capacity,
                2,
                config.local_layer_count,
                page_size,
                kv_heads,
                config.head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        local = self.storage.permute(1, 2, 0, 3, 4, 5)
        self.local_keys, self.local_values = local[0], local[1]
        packing = config.history_blocks_per_parent
        global_shape = (
            physical_parent_capacity * packing,
            2,
            config.global_layer_count,
            page_size,
            kv_heads,
            config.head_dim,
        )
        global_view = self.storage.view(global_shape).permute(1, 2, 0, 3, 4, 5)
        self.global_keys, self.global_values = global_view[0], global_view[1]
        self.packing = packing
        self.sliding_window = config.sliding_window

    def cache_storage_buffers(self, groups, *, prefix_length):
        buffers = []
        for group, (first, addresses) in enumerate(groups):
            keys, values = (
                (self.global_keys, self.global_values)
                if group == FULL_CACHE_GROUP
                else (self.local_keys, self.local_values)
            )
            for block, (slab, slot) in enumerate(addresses, first):
                rows = live_rows(
                    prefix_length,
                    block,
                    self.page_size,
                    None if group == FULL_CACHE_GROUP else self.sliding_window,
                )
                page = slab * self.packing + slot if group == FULL_CACHE_GROUP else slab
                buffers.append(
                    byte_regions(
                        {
                            f"{name}/{layer}": tensor[layer, page, rows]
                            for name, tensor in (("keys", keys), ("values", values))
                            for layer in range(keys.shape[0])
                        }
                    )
                )
        return tuple(buffers)


@dataclass(frozen=True, slots=True)
class _DevicePagedPlan:
    qo_indptr: Tensor
    kv_indptr: Tensor
    page_indices: Tensor
    last_page_lengths: Tensor
    append_pages: Tensor
    append_offsets: Tensor


LogitsObserver = Callable[["BatchPlan", Tensor], None]


class RippleCudaRunner(CudaRunner):
    """Paged BF16 Ripple forward pass with eager or CUDA-graph decode.

    ``num_layers`` runs the first layers of the checkpoint at full width, e.g.
    ``num_layers=5`` is one ``[S, S, S, S, G]`` period (about 8.3 GB of
    weights), the same code path as the 80-layer model.

    ``logits_observer(plan, logits)`` is called after every forward with the
    last-token logits ``[num_sequences, vocab]`` before sampling. Graph replay
    reuses its output buffer, so observers must copy what they keep.
    """

    max_num_seqs = DEFAULT_MAX_NUM_SEQS
    parallelism_capabilities = ParallelismCapabilities(attn_tp=True, expert_tp=True)

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        num_layers: int | None = None,
        cache_parent_capacity: int | None = None,
        page_size: int = PAGE_SIZE,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        device: str | torch.device,
        kv_cache_dtype: str = "bfloat16",
        cuda_graph_prefill: str = "breakable",
        cuda_graph_decode: str = "full",
        cuda_graph_decode_shapes: object | None = None,
        static_memory_fraction: float = 0.95,
        parallel_groups: ParallelGroups | None = None,
    ) -> None:
        for name, value in (
            ("prefill_chunk_size", prefill_chunk_size),
            ("page_size", page_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if kv_cache_dtype != "bfloat16":
            raise ValueError("the Ripple CUDA runner stores BF16 KV only")
        if cache_parent_capacity is not None and (
            isinstance(cache_parent_capacity, bool)
            or not isinstance(cache_parent_capacity, int)
            or cache_parent_capacity <= 0
        ):
            raise ValueError("cache_parent_capacity must be a positive integer")
        self.page_size = page_size
        self.prefill_chunk_size = prefill_chunk_size
        if parallel_groups is not None:
            self.parallelism_capabilities.validate(
                parallel_groups.topology, model="Ripple"
            )
        self.parallel_groups = parallel_groups
        self.attention_parallel = TensorParallelContext.from_group(
            None if parallel_groups is None else parallel_groups.attn_tp
        )
        self.expert_parallel = TensorParallelContext.from_group(
            None if parallel_groups is None else parallel_groups.expert_tp
        )
        super().__init__(
            device=device,
            label="Ripple",
            kv_cache_dtype=kv_cache_dtype,
            cuda_graph_prefill=cuda_graph_prefill,
            cuda_graph_decode=cuda_graph_decode,
        )
        capability = torch.cuda.get_device_capability(self.device)
        if capability < (8, 0):
            raise RuntimeError(
                "Ripple requires an Ampere (SM80) or newer GPU, "
                f"got sm{capability[0]}{capability[1]}"
            )
        self.logits_observer: LogitsObserver | None = None

        self.model_path = resolve_model_path(model, revision)
        self._model, self.config, self.raw_config = load_ripple_cuda_model(
            self.model_path,
            device=self.device,
            num_layers=num_layers,
            tensor_parallel=self.expert_parallel,
            attention_parallel=self.attention_parallel,
        )
        self.geometry = attention_geometry(self.config, self.attention_parallel.layout)
        self.sliding_window = self.config.sliding_window
        self.window_retention_tokens = self.prefill_chunk_size + self.sliding_window - 1
        self.window_graph_page_capacity = (
            1 + (self.window_retention_tokens - 1 + page_size - 1) // page_size
        )
        self._configure_parallel_graphs(self._model, parallel_groups)
        self._graph_attention: dict[GraphShape, tuple[Any, Any]] = {}
        self._graph_logits: dict[GraphShape, Tensor] = {}
        # Fix the model's static bytes before sizing the cache from free memory.
        self._model_memory_bytes = self._count_model_memory()
        self.model_load_peak_bytes = torch.cuda.max_memory_allocated(self.device)

        torch.cuda.synchronize(self.device)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        capacity_plan = plan_ripple_kv_capacity(
            total_memory,
            free_memory,
            static_memory_fraction,
            config=self.config,
            page_size=page_size,
            prefill_chunk_size=self.prefill_chunk_size,
            cache_element_bytes=self.kv_cache_torch_dtype.itemsize,
            tensor_parallel_size=self.attention_parallel.size,
        )
        local_capacity = capacity_plan.parent_capacity
        if cache_parent_capacity is not None:
            local_capacity = min(local_capacity, cache_parent_capacity)
        local_capacity = replicated_cache_capacity(
            local_capacity, None if parallel_groups is None else parallel_groups.replica
        )
        self.cache_parent_capacity = local_capacity
        self.cache_capacity_plan = replace(
            capacity_plan,
            parent_capacity=local_capacity,
            allocation_bytes=(local_capacity + 1) * capacity_plan.bytes_per_parent,
        )

        maximum_history_pages = (
            self.config.max_position_embeddings + page_size - 1
        ) // page_size
        decode_graph_shapes = resolve_decode_graph_shapes(
            cuda_graph_decode_shapes,
            policy=self.cuda_graph_decode,
            width=min(
                maximum_history_pages,
                self.cache_parent_capacity * self.config.history_blocks_per_parent,
            ),
            speculative=False,
            max_batch_size=self.max_num_seqs,
        )
        if any(batch > self.max_num_seqs for batch, _ in decode_graph_shapes):
            raise ValueError(
                f"Ripple CUDA graph batch capacity exceeds {self.max_num_seqs}"
            )
        if any(width > maximum_history_pages for _, width in decode_graph_shapes):
            raise ValueError("Ripple CUDA graph page width exceeds context length")
        self._configure_cuda_graphs(
            {
                "decode": CudaGraphMode(
                    decode_graph_shapes,
                    self._new_decode_graph_step,
                    lambda shape: self._graph_capture_inputs(
                        shape, self.config.vocab_size
                    ),
                )
            }
        )

        self.cache_layout = ripple_cache_layout(
            self.config,
            parent_capacity=self.cache_parent_capacity,
            block_tokens=page_size,
            prefill_chunk_size=self.prefill_chunk_size,
        )
        self.runner_config = RunnerConfig(
            max_batch_tokens=self.prefill_chunk_size,
            prefill_chunk_size=self.prefill_chunk_size,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
        )
        self._configure_cache_binding(
            CachePoolBinding(self.cache_layout, resource="Ripple KV")
        )
        self._cache = _RippleKvArena(
            self.device,
            self.config,
            self.cache_parent_capacity,
            kv_heads=self.geometry.kv_heads,
            page_size=page_size,
            dtype=self.kv_cache_torch_dtype,
        )
        self._graph_attention_workspaces = tuple(
            torch.zeros(_GRAPH_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
            for _ in range(2)
        )
        self._build_attention_wrappers()
        self._prefill_feed_forward = self._configure_feed_forward_graphs()

    # -- construction helpers -------------------------------------------------

    def _count_model_memory(self) -> int:
        tensors = list(self._model.parameters()) + list(self._model.buffers())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _configure_feed_forward_graphs(self):
        return self._configure_prefill_graphs(
            tuple(
                lambda hidden, _ids, layer=layer: layer.feed_forward(hidden)
                for layer in self._model.model.layers
            ),
            hidden_size=self.config.hidden_size,
            max_tokens=self.prefill_chunk_size,
        )

    @property
    def model(self) -> RippleCudaForCausalLM:
        return self._model

    @property
    def model_memory_bytes(self) -> int:
        return self._model_memory_bytes

    def _build_attention_wrappers(self) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        def workspace() -> Tensor:
            size = _prefill_workspace_bytes(
                self.prefill_chunk_size, self.geometry.q_heads, self.config.head_dim
            )
            return torch.zeros(size, dtype=torch.uint8, device=self.device)

        window_workspace, full_workspace = workspace(), workspace()
        self._workspaces = (window_workspace, full_workspace)
        self._window_prefill = BatchPrefillWithPagedKVCacheWrapper(
            window_workspace, "NHD", backend="auto"
        )
        self._full_prefill = BatchPrefillWithPagedKVCacheWrapper(
            full_workspace, "NHD", backend="auto"
        )
        # Tensor-core decode serves the 10:1 GQA group and matches the graph path.
        self._window_decode = BatchDecodeWithPagedKVCacheWrapper(
            window_workspace, "NHD", use_tensor_cores=True, backend="auto"
        )
        self._full_decode = BatchDecodeWithPagedKVCacheWrapper(
            full_workspace, "NHD", use_tensor_cores=True, backend="auto"
        )

    def release(self, request_ids: list[int]) -> None:
        """KV lives in Engine-leased pages; there is no per-request runner state."""

    # -- attention planning ---------------------------------------------------

    def _device_plan(self, plan: PagedAttentionPlan) -> _DevicePagedPlan:
        def tensor(values, dtype):
            return torch.as_tensor(values, dtype=dtype, device=self.device)

        append_slots = tensor(plan.append_slots, torch.int64)
        return _DevicePagedPlan(
            qo_indptr=tensor(plan.qo_indptr, torch.int32),
            kv_indptr=tensor(plan.kv_indptr, torch.int32),
            page_indices=tensor(plan.page_indices, torch.int32),
            last_page_lengths=tensor(plan.last_page_lengths, torch.int32),
            append_pages=torch.div(append_slots, self.page_size, rounding_mode="floor"),
            append_offsets=append_slots.remainder(self.page_size),
        )

    @property
    def _sm_scale(self) -> float:
        # Standard softmax scale; Q is not pre-scaled (Gemma 4 uses 1.0).
        return self.config.head_dim**-0.5

    def _plan_attention(
        self, batch: RippleBatchPlan, window: _DevicePagedPlan, full: _DevicePagedPlan
    ) -> tuple[Any, Any]:
        geometry, head_dim = self.geometry, self.config.head_dim
        common = {
            "pos_encoding_mode": "NONE",  # RoPE is applied before the cache write
            "q_data_type": torch.bfloat16,
            "kv_data_type": self._cache.dtype,
            "o_data_type": torch.bfloat16,
            "sm_scale": self._sm_scale,
        }
        if batch.phase == BatchPhase.PREFILL:
            for wrapper, plan, left in (
                (self._window_prefill, window, self.sliding_window - 1),
                (self._full_prefill, full, -1),
            ):
                wrapper.plan(
                    plan.qo_indptr,
                    plan.kv_indptr,
                    plan.page_indices,
                    plan.last_page_lengths,
                    geometry.q_heads,
                    geometry.kv_heads,
                    head_dim,
                    self.page_size,
                    causal=True,
                    window_left=left,
                    **common,
                )
            return self._window_prefill, self._full_prefill
        for wrapper, plan, left in (
            (self._window_decode, window, self.sliding_window - 1),
            (self._full_decode, full, -1),
        ):
            wrapper.plan(
                plan.kv_indptr,
                plan.page_indices,
                plan.last_page_lengths,
                geometry.q_heads,
                geometry.kv_heads,
                head_dim,
                self.page_size,
                window_left=left,
                **common,
            )
        return self._window_decode, self._full_decode

    def _run_transformer(
        self,
        hidden: Tensor,
        positions: Tensor,
        window: _DevicePagedPlan,
        full: _DevicePagedPlan,
        window_wrapper: Any,
        full_wrapper: Any,
    ) -> Tensor:
        cache = self._cache

        def attend(layer: int, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
            group, slot = self.config.layer_cache_slots[layer]
            if group == WINDOW_CACHE_GROUP:
                keys, values = cache.local_keys[slot], cache.local_values[slot]
                plan, wrapper = window, window_wrapper
            else:
                keys, values = cache.global_keys[slot], cache.global_values[slot]
                plan, wrapper = full, full_wrapper
            keys[plan.append_pages, plan.append_offsets] = key.to(cache.dtype)
            values[plan.append_pages, plan.append_offsets] = value.to(cache.dtype)
            return wrapper.run(query, (keys, values))

        return self._model.model(hidden, positions, attend, self._prefill_feed_forward)

    # -- CUDA graph decode ----------------------------------------------------

    def _graph_capture_inputs(
        self, shape: GraphShape, vocab_size: int
    ) -> dict[str, object]:
        """Synthetic decode rows backed only by graph-reserved parent zero."""

        batch_size, history_width = shape
        inputs: dict[str, object] = {
            "token_ids": (0,) * batch_size,
            "positions": (0,) * batch_size,
            **capture_sampling_inputs(batch_size, vocab_size),
        }
        for name, page_width in (
            ("window", self.window_graph_page_capacity),
            ("full", history_width),
        ):
            inputs.update(
                {
                    f"{name}_kv_indptr": tuple(range(batch_size + 1)),
                    f"{name}_page_indices": (0,) * (batch_size * page_width),
                    f"{name}_last_page_lengths": (1,) * batch_size,
                    f"{name}_append_pages": (0,) * batch_size,
                    f"{name}_append_offsets": tuple(
                        row % self.page_size for row in range(batch_size)
                    ),
                }
            )
        return inputs

    def _before_cuda_graph_capture(self) -> None:
        # Capture reads parent zero; keep it finite so eager/replay agree.
        self._cache.storage[0].zero_()

    def _after_cuda_graph_capture(self) -> None:
        self._cache.storage[0].zero_()
        torch.cuda.synchronize(self.device)

    def _plan_graph_attention(
        self, wrapper, indptr, indices, last, *, window: bool, non_blocking=False
    ) -> None:
        """One FlashInfer decode contract shared by capture and replay."""

        wrapper.plan(
            indptr,
            indices,
            last,
            self.geometry.q_heads,
            self.geometry.kv_heads,
            self.config.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=self._cache.dtype,
            o_data_type=torch.bfloat16,
            sm_scale=self._sm_scale,
            window_left=self.sliding_window - 1 if window else -1,
            disable_split_kv=True,
            non_blocking=non_blocking,
        )

    def _new_decode_graph_step(self, shape: GraphShape) -> CudaGraphStep:
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        batch_size, history_width = shape

        def empty(dtype, *dimensions):
            return torch.empty(dimensions, dtype=dtype, device=self.device)

        def attention(width: int, *, window: bool):
            workspace = self._graph_attention_workspaces[0 if window else 1]
            kv_indptr = empty(torch.int32, batch_size + 1)
            page_indices = empty(torch.int32, batch_size * width)
            last_page_lengths = empty(torch.int32, batch_size)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=kv_indptr,
                paged_kv_indices_buffer=page_indices,
                paged_kv_last_page_len_buffer=last_page_lengths,
                backend="fa2",
            )
            self._plan_graph_attention(
                wrapper,
                torch.arange(batch_size + 1, dtype=torch.int32, device=self.device),
                torch.zeros(batch_size, dtype=torch.int32, device=self.device),
                torch.ones(batch_size, dtype=torch.int32, device=self.device),
                window=window,
            )
            plan = _DevicePagedPlan(
                qo_indptr=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=self.device
                ),
                kv_indptr=kv_indptr,
                page_indices=page_indices,
                last_page_lengths=last_page_lengths,
                append_pages=empty(torch.int64, batch_size),
                append_offsets=empty(torch.int64, batch_size),
            )
            return wrapper, plan

        window_wrapper, window = attention(self.window_graph_page_capacity, window=True)
        full_wrapper, full = attention(history_width, window=False)
        self._graph_attention[shape] = (window_wrapper, full_wrapper)
        token_ids = empty(torch.int64, batch_size)
        positions = empty(torch.int64, batch_size)
        output_ids = empty(torch.int64, batch_size)
        sampling = self._sampler.empty(batch_size, self.device)

        def launch() -> None:
            hidden = self._model.embed_tokens(token_ids)
            hidden = self._run_transformer(
                hidden, positions, window, full, window_wrapper, full_wrapper
            )
            logits = self._model.logits(hidden)
            self._graph_logits[shape] = logits
            self._sampler.sample_tensors(logits, sampling, out=output_ids)

        inputs: dict[str, Any] = {"token_ids": token_ids, "positions": positions}
        for name, plan in (("window", window), ("full", full)):
            inputs.update(
                {
                    f"{name}_kv_indptr": plan.kv_indptr,
                    f"{name}_page_indices": plan.page_indices,
                    f"{name}_last_page_lengths": plan.last_page_lengths,
                    f"{name}_append_pages": plan.append_pages,
                    f"{name}_append_offsets": plan.append_offsets,
                }
            )
        inputs.update(self._sampler.graph_inputs(sampling))
        return CudaGraphStep(inputs=inputs, output=output_ids, launch=launch)

    def _prepare_graph_attention(
        self, shape: GraphShape, inputs: dict[str, object]
    ) -> None:
        def tensor(values):
            return torch.as_tensor(values, dtype=torch.int32, device=self.device)

        for window, wrapper in zip((True, False), self._graph_attention[shape]):
            name = "window" if window else "full"
            indptr = inputs[f"{name}_kv_indptr"]
            page_count = int(np.asarray(indptr)[-1])
            self._plan_graph_attention(
                wrapper,
                tensor(indptr),
                tensor(inputs[f"{name}_page_indices"])[:page_count],
                tensor(inputs[f"{name}_last_page_lengths"]),
                window=window,
            )

    def _graph_page_inputs(
        self,
        name: str,
        plan: PagedAttentionPlan,
        *,
        active_rows: int,
        graph_rows: int,
        width: int,
    ) -> dict[str, object]:
        row_widths = np.diff(plan.kv_indptr)
        if len(row_widths) != active_rows or np.any(row_widths > width):
            raise GraphInputError(f"{name} page table exceeds CUDA graph shape")
        indices = np.zeros(graph_rows * width, dtype=np.int32)
        indptr = np.zeros(graph_rows + 1, dtype=np.int32)
        cursor = 0
        for row in range(graph_rows):
            if row < active_rows:
                start, end = map(int, plan.kv_indptr[row : row + 2])
                count = end - start
                indices[cursor : cursor + count] = plan.page_indices[start:end]
            else:
                count = 1  # padding rows read parent zero
            cursor += count
            indptr[row + 1] = cursor
        last_page_lengths = np.ones(graph_rows, dtype=np.int32)
        last_page_lengths[:active_rows] = plan.last_page_lengths
        append_pages = np.zeros(graph_rows, dtype=np.int64)
        append_offsets = np.arange(graph_rows, dtype=np.int64) % self.page_size
        append_pages[:active_rows] = plan.append_slots // self.page_size
        append_offsets[:active_rows] = plan.append_slots % self.page_size
        return {
            f"{name}_kv_indptr": indptr,
            f"{name}_page_indices": indices,
            f"{name}_last_page_lengths": last_page_lengths,
            f"{name}_append_pages": append_pages,
            f"{name}_append_offsets": append_offsets,
        }

    def _graph_forward(
        self, native: BatchPlan, batch: RippleBatchPlan
    ) -> ForwardOutput | None:
        if not self._cuda_graph_mode_enabled("decode"):
            return None
        active_rows = native.num_sequences
        history_width = int(np.diff(batch.full.kv_indptr).max())
        window_width = int(np.diff(batch.window.kv_indptr).max())
        if window_width > self.window_graph_page_capacity:
            raise GraphInputError("window page table exceeds CUDA graph shape")
        shape = self._select_cuda_graph(
            "decode",
            match=lambda candidate: (
                candidate[0] >= active_rows and candidate[1] >= history_width
            ),
        )
        if shape is None:
            return None
        graph_rows, graph_history_width = shape
        padding_rows = graph_rows - active_rows
        inputs: dict[str, object] = {
            "token_ids": np.pad(batch.token_ids, (0, padding_rows)),
            "positions": np.pad(batch.positions, (0, padding_rows)),
        }
        inputs.update(
            self._graph_page_inputs(
                "window",
                batch.window,
                active_rows=active_rows,
                graph_rows=graph_rows,
                width=self.window_graph_page_capacity,
            )
        )
        inputs.update(
            self._graph_page_inputs(
                "full",
                batch.full,
                active_rows=active_rows,
                graph_rows=graph_rows,
                width=graph_history_width,
            )
        )
        inputs.update(
            pad_sampling_inputs(
                self._sampler, native, self.config.vocab_size, padding_rows
            )
        )
        self._prepare_graph_attention(shape, inputs)
        output = self._run_cuda_graph("decode", shape, inputs)
        logits = self._graph_logits[shape]
        if self.logits_observer is not None:
            self.logits_observer(native, logits[:active_rows])
        if native.needs_logprobs:
            return format_output(logits, output, native)
        return token_output(output[:active_rows])

    # -- eager execution ------------------------------------------------------

    def _eager_transformer(self, batch: RippleBatchPlan) -> Tensor:
        token_ids = torch.as_tensor(
            batch.token_ids, dtype=torch.int64, device=self.device
        )
        positions = torch.as_tensor(
            batch.positions, dtype=torch.int64, device=self.device
        )
        hidden = self._model.embed_tokens(token_ids)
        window = self._device_plan(batch.window)
        full = self._device_plan(batch.full)
        window_wrapper, full_wrapper = self._plan_attention(batch, window, full)
        with (
            self._prefill_graphs()
            if batch.phase == BatchPhase.PREFILL
            else nullcontext()
        ):
            return self._run_transformer(
                hidden, positions, window, full, window_wrapper, full_wrapper
            )

    def _sample_target(self, logits: Tensor, plan: BatchPlan, step: int) -> Tensor:
        return self._sampler(logits, plan, offset_delta=step * SAMPLING_OFFSET_STRIDE)

    @torch.inference_mode()
    @observe_execution
    def forward(self, plan: BatchPlan) -> ForwardOutput:
        self._check_cuda_graphs(plan.phase)
        batch = build_ripple_batch_plan(plan, self.cache_layout)
        with self._cuda_failure:
            if batch.phase == BatchPhase.DECODE:
                graph_output = self._graph_forward(plan, batch)
                if graph_output is not None:
                    return graph_output
            hidden = self._eager_transformer(batch)
            last = torch.as_tensor(
                batch.last_token_indices, dtype=torch.int64, device=self.device
            )
            logits = self._model.logits(hidden[last])
            if self.logits_observer is not None:
                self.logits_observer(plan, logits)
            scores = prompt_scores(plan, hidden, self._model.logits)
            return format_output(
                logits,
                self._sample_target(logits, plan, 0),
                plan,
                prompt_logprobs=scores,
            )


__all__ = [
    "PAGE_SIZE",
    "PREFILL_CHUNK_SIZE",
    "RippleCudaRunner",
    "plan_ripple_kv_capacity",
]
