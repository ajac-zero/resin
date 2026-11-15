"""Resolve, shard, and prepare a Ripple (``ricursive``) BF16 checkpoint."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from resin.backend.cuda.parallel import TensorParallelContext
from resin.backend.loading import (
    MODEL_FILES,
    assign_model_tensor,
    read_config,
    resolve_model,
)
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.models.ripple.config import RippleConfig, attention_geometry
from resin_models_cuda.ripple.model import RippleCudaForCausalLM
from resin_models_cuda.ripple.parallel import prepare_ripple_tp

DEFAULT_MODEL = "ricursiveAI/Ripple-65B-Preview"
DEFAULT_REVISION = "b172323ffd63e62de9ce1ba3830dfffe493c7671"

_LAYER_KEY = re.compile(r"model\.layers\.(\d+)\.")


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=list(MODEL_FILES)
        )
    )


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    selected = (
        DEFAULT_REVISION
        if revision is None and str(model) == DEFAULT_MODEL
        else revision
    )
    return resolve_model(model, selected, _download_model)


def _tensor_files(root: Path) -> dict[str, Path]:
    """Map tensor names to shard paths without requiring unused shards to exist."""

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        return indexed_tensor_files(root)
    weight_map = json.loads(index_path.read_text()).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path.name} must contain a non-empty weight_map")
    files = {}
    for key, filename in weight_map.items():
        relative = Path(str(filename))
        if len(relative.parts) != 1 or relative.suffix != ".safetensors":
            raise ValueError(f"invalid checkpoint shard name: {filename!r}")
        files[key] = root / relative
    return files


def load_ripple_cuda_model(
    model_path: str | Path,
    *,
    device: str | torch.device,
    num_layers: int | None = None,
    tensor_parallel: TensorParallelContext | None = None,
    attention_parallel: TensorParallelContext | None = None,
) -> tuple[RippleCudaForCausalLM, RippleConfig, dict[str, Any]]:
    """Load Ripple, or its first ``num_layers`` layers, as a BF16 inference model.

    A reduced load needs only the embeddings, final norm, LM head and the first
    layers, so only the shard files holding those tensors are opened (about
    8.3 GB for ``num_layers=5`` instead of the 130 GB checkpoint). Tensors are
    read and tensor-parallel sharded on the CPU; the only device transfer moves
    the rank-local weights, so no rank ever holds unsharded weights on the GPU.
    """

    root = Path(model_path).expanduser().resolve()
    raw_config = read_config(root)
    if raw_config.get("quantization") or raw_config.get("quantization_config"):
        raise ValueError("the Ripple CUDA runner requires an unquantized checkpoint")
    config = RippleConfig.from_dict(raw_config)
    if num_layers is not None:
        config = config.same_shape_reduced(num_layers)
    if attention_parallel is not None:
        attention_geometry(config, attention_parallel.layout)
    if tensor_parallel is not None:
        attention_geometry(config, tensor_parallel.layout)
    target = torch.device(device)

    with torch.device("meta"):
        model = RippleCudaForCausalLM(config)
    tensor_files = _tensor_files(root)
    expected = set(model.state_dict())
    if missing := expected - tensor_files.keys():
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:8]}")
    unexpected = {
        key
        for key in tensor_files.keys() - expected
        if not (
            (layer := _LAYER_KEY.match(key))
            and int(layer.group(1)) >= config.num_hidden_layers
        )
    }
    if unexpected:
        raise ValueError(f"checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")

    by_file: dict[Path, list[str]] = {}
    for key in expected:
        by_file.setdefault(tensor_files[key], []).append(key)
    for path in by_file:
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {path}")
    with SafetensorReader() as reader:
        for path, keys in by_file.items():
            for key in keys:
                assign_model_tensor(
                    model,
                    key,
                    reader.read(path, key),
                    device="cpu",
                    dtype=torch.bfloat16,
                )
    # Recreate nonpersistent buffers before moving the loaded model: only
    # rank-local weights should ever occupy device memory.
    model.materialize_buffers("cpu")
    prepared = prepare_ripple_tp(
        model, attention_parallel=attention_parallel, tensor_parallel=tensor_parallel
    )
    prepared = prepared.to(target)
    model.pack_projections()
    model.eval()
    return model, config, raw_config


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "load_ripple_cuda_model",
    "resolve_model_path",
]
