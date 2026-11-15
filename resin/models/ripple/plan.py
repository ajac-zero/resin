"""Hybrid KV cache layout and paged-attention plans for Ripple."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from resin import BatchPhase, CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.plans import CacheGroupView, PagedCachePlan
from resin.models.gemma4.plan import PagedAttentionPlan
from resin.models.ripple.config import (
    FULL_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    RippleConfig,
)


@dataclass(frozen=True, slots=True)
class RippleBatchPlan:
    phase: BatchPhase
    token_ids: npt.NDArray[np.int64]
    positions: npt.NDArray[np.int64]
    token_indptr: npt.NDArray[np.int32]
    full: PagedAttentionPlan
    window: PagedAttentionPlan
    last_token_indices: npt.NDArray[np.int64]


def ripple_cache_layout(
    config: RippleConfig,
    *,
    parent_capacity: int,
    block_tokens: int,
    prefill_chunk_size: int,
) -> CacheLayout:
    """Group 0 holds full-attention history; group 1 a sliding window.

    A window row may be written ``prefill_chunk_size`` tokens at a time while
    still reading ``sliding_window`` tokens, so it retains their sum minus one.
    """

    if prefill_chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be positive")
    return CacheLayout(
        parent_capacity=parent_capacity,
        groups=[
            CacheGroupLayout(
                block_tokens, blocks_per_slab=config.history_blocks_per_parent
            ),
            CacheGroupLayout(
                block_tokens,
                retention=CacheRetention.WINDOW,
                window_tokens=prefill_chunk_size + config.sliding_window - 1,
                read_window_tokens=config.sliding_window,
            ),
        ],
    )


def _paged_plan(
    group: CacheGroupView,
    token_indptr: npt.NDArray[np.int32],
    positions: npt.NDArray[np.int64],
    page_size: int,
) -> PagedAttentionPlan:
    packed_blocks = group.packed_block_ids.astype(np.int64, copy=False)
    page_rows, last_page_lengths, first_positions, kv_lengths = [], [], [], []
    for row in range(len(token_indptr) - 1):
        end = int(token_indptr[row + 1])
        post_length = int(positions[end - 1]) + 1
        first_block = int(group.block_table_start_indices[row])
        pages = packed_blocks[group.block_slice(row)]
        # A decode lease can reserve pages beyond the current position.
        required = (post_length - 1) // page_size - first_block + 1
        if required <= 0 or required > len(pages):
            raise ValueError("Ripple paged attention row does not match its lease")
        page_rows.append(pages[:required])
        last_page_lengths.append((post_length - 1) % page_size + 1)
        first_positions.append(first_block * page_size)
        kv_lengths.append(post_length - first_block * page_size)
    kv_indptr = np.zeros(len(page_rows) + 1, dtype=np.int32)
    kv_indptr[1:] = np.cumsum([len(row) for row in page_rows], dtype=np.int32)
    return PagedAttentionPlan(
        qo_indptr=token_indptr,
        kv_indptr=kv_indptr,
        page_indices=np.concatenate(page_rows).astype(np.int32, copy=False),
        last_page_lengths=np.asarray(last_page_lengths, dtype=np.int32),
        append_slots=(
            group.packed_append_ids.astype(np.int64) * page_size
            + group.append_block_offsets
        ).astype(np.int64, copy=False),
        first_token_positions=tuple(first_positions),
        kv_lengths=tuple(kv_lengths),
    )


def build_ripple_batch_plan(plan, cache_layout: CacheLayout) -> RippleBatchPlan:
    """Validate an Engine ``BatchPlan`` and translate it to Ripple page tables."""

    if cache_layout.group_count != 2:
        raise ValueError("Ripple cache layouts have a full and a window group")
    cache = PagedCachePlan.from_plan(plan, cache_layout)
    batch = cache.batch
    if batch.phase not in (BatchPhase.PREFILL, BatchPhase.DECODE):
        raise ValueError("plan phase must be prefill or decode")
    page_size = cache_layout.groups[FULL_CACHE_GROUP].block_tokens
    if cache_layout.groups[WINDOW_CACHE_GROUP].block_tokens != page_size:
        raise ValueError("Ripple cache groups must share one block size")
    token_indptr = batch.token_indptr.astype(np.int32, copy=False)
    positions = batch.positions.astype(np.int64, copy=False)
    return RippleBatchPlan(
        phase=batch.phase,
        token_ids=batch.token_ids.astype(np.int64, copy=False),
        positions=positions,
        token_indptr=token_indptr,
        full=_paged_plan(
            cache.groups[FULL_CACHE_GROUP], token_indptr, positions, page_size
        ),
        window=_paged_plan(
            cache.groups[WINDOW_CACHE_GROUP], token_indptr, positions, page_size
        ),
        last_token_indices=(token_indptr[1:] - 1).astype(np.int64),
    )


__all__ = ["RippleBatchPlan", "build_ripple_batch_plan", "ripple_cache_layout"]
