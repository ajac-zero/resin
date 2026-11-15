"""Ripple (``ricursive``) BF16 CUDA support for Ampere and newer GPUs.

Importing this package needs neither CUDA hardware nor ``flashinfer`` or
``triton``: kernels are imported when a runner is constructed.
"""

from resin_models_cuda.ripple.checkpoint import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    load_ripple_cuda_model,
)
from resin_models_cuda.ripple.cuda import RippleCudaRunner, plan_ripple_kv_capacity
from resin_models_cuda.ripple.model import RippleCudaForCausalLM

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "RippleCudaForCausalLM",
    "RippleCudaRunner",
    "load_ripple_cuda_model",
    "plan_ripple_kv_capacity",
]
