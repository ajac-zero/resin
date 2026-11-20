"""BF16 arithmetic, mutable CUDA graphs, and reserved-page metadata contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from resin_models_cuda.gemma4.edge_kernels import (
    gate_up_gelu_m1,
    kv_append,
    gelu_tanh_mul,
    linear_m1,
    paged_decode_q1,
    linear_gelu,
    norm_residual,
    sparse_gather_dot,
    strided_qk_rmsnorm_rope,
    update_reserved_paged_metadata,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _capture(function):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def _assert_replay(graph, captured, expected):
    graph.replay()
    torch.cuda.synchronize()
    if isinstance(expected, tuple):
        for actual, wanted in zip(captured, expected, strict=True):
            if wanted is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
    else:
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


def _compilations(monkeypatch, kernel):
    from triton import knobs

    compiled = []
    previous = knobs.runtime.jit_post_compile_hook

    def post_compile(**event):
        if event["fn"].name == kernel:
            compiled.append(str(event["key"]))
        if previous is not None:
            return previous(**event)

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", post_compile)
    return compiled


def _discrete_storage(*shape):
    powers = torch.randint(-2, 3, shape, device="cuda").float().exp2()
    signs = torch.randint(0, 2, shape, device="cuda").float() * 2 - 1
    return (powers * signs).bfloat16()


def _norm(hidden, weight, eps):
    values = hidden.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return ((values * inverse) * weight.float()).bfloat16()


def _rope_reference(hidden, weight, cos, sin, eps):
    normalized = _norm(hidden, weight, eps)
    left, right = normalized.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    # Separate BF16 products and addition expose accidental FMA/round changes.
    return normalized * cos[:, None, :] + rotated * sin[:, None, :]


def _qk_inputs(tokens, width, with_key):
    storage = _discrete_storage(tokens, 11, width + 16)
    frequencies = torch.randn(tokens, width // 2, device="cuda")
    if width == 512:
        frequencies[:, width // 8 :] = 0  # Global attention rotates a partial head.
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return (
        storage[:, :8, 3 : 3 + width],
        storage[:, 8:9, 3 : 3 + width] if with_key else None,
        torch.randn(width, dtype=torch.bfloat16, device="cuda"),
        torch.randn(width, dtype=torch.bfloat16, device="cuda") if with_key else None,
        frequencies.cos().bfloat16(),
        frequencies.sin().bfloat16(),
    )


@pytest.mark.parametrize("width", [256, 512])
@pytest.mark.parametrize("with_key", [False, True])
@torch.inference_mode()
def test_strided_norm_rope_rounding_replay_and_request_values(
    width, with_key, monkeypatch
):
    torch.manual_seed(71)
    inputs = _qk_inputs(3, width, with_key)

    def operation():
        return strided_qk_rmsnorm_rope(*inputs, eps=1e-6)

    graph, captured = _capture(operation)
    compiled = _compilations(monkeypatch, "_strided_qk_rmsnorm_rope")
    for _ in range(3):
        for value in inputs:
            if value is not None:
                value.copy_(torch.randn_like(value))
        _assert_replay(graph, captured, operation())
    for tokens, eps in [(1, 3e-6), (65, 1e-5), (513, 1e-7)]:
        changed = _qk_inputs(tokens, width, with_key)
        query, key, qw, kw, cos, sin = changed
        actual_q, actual_k = strided_qk_rmsnorm_rope(*changed, eps=eps)
        torch.testing.assert_close(
            actual_q, _rope_reference(query, qw, cos, sin, eps), atol=0, rtol=0
        )
        if key is None:
            assert actual_k is None
        else:
            torch.testing.assert_close(
                actual_k, _rope_reference(key, kw, cos, sin, eps), atol=0, rtol=0
            )
    assert compiled == [], f"ordinary request values caused compilation: {compiled}"


def _residual_inputs(tokens, scaled):
    storage = _discrete_storage(tokens, 1600)
    return (
        storage[:, 3:1539],
        torch.randn(1536, dtype=torch.bfloat16, device="cuda"),
        torch.randn_like(storage)[:, 7:1543],
        torch.tensor([1.125], dtype=torch.bfloat16, device="cuda") if scaled else None,
    )


def _residual_reference(hidden, weight, residual, scale, eps):
    added = _norm(hidden, weight, eps) + residual
    return added if scale is None else added * scale


@pytest.mark.parametrize("scaled", [False, True])
@torch.inference_mode()
def test_post_norm_residual_bf16_rounding_replay_and_request_values(
    scaled, monkeypatch
):
    torch.manual_seed(83)
    hidden, weight, residual, scale = _residual_inputs(3, scaled)

    def operation():
        return norm_residual(hidden, weight, residual, scale=scale, eps=1e-6)

    graph, captured = _capture(operation)
    compiled = _compilations(monkeypatch, "_norm_residual")
    for iteration in range(3):
        h, w, r, _ = _residual_inputs(3, scaled)
        hidden.copy_(h)
        weight.copy_(w)
        residual.copy_(r)
        if scale is not None:
            scale.fill_(0.75 + iteration / 8)
        expected = _residual_reference(hidden, weight, residual, scale, 1e-6)
        _assert_replay(graph, captured, expected)
    for tokens, eps in [(1, 3e-6), (65, 1e-5), (513, 1e-7)]:
        h, w, r, s = _residual_inputs(tokens, scaled)
        torch.testing.assert_close(
            norm_residual(h, w, r, scale=s, eps=eps),
            _residual_reference(h, w, r, s, eps),
            atol=0,
            rtol=0,
        )
    assert compiled == [], f"ordinary request values caused compilation: {compiled}"


@pytest.mark.parametrize("width", [127, 256])
@torch.inference_mode()
def test_sparse_head_fp64_oracle_and_changed_input_graph(width):
    torch.manual_seed(29)
    hidden = torch.randn(3, width, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(1024, width, dtype=hidden.dtype, device="cuda")
    indices = torch.randint(1024, (3, 74), device="cuda")[:, ::2]

    def operation():
        return sparse_gather_dot(hidden, weights, indices)

    graph, captured = _capture(operation)
    for _ in range(3):
        hidden.copy_(torch.randn_like(hidden))
        weights.copy_(torch.randn_like(weights))
        indices.copy_(torch.randint(1024, indices.shape, device="cuda"))
        expected = (
            (hidden.double()[:, None, :] * weights[indices].double()).sum(-1).bfloat16()
        )
        actual = operation()
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        _assert_replay(graph, captured, actual)


@torch.inference_mode()
def test_cublaslt_gelu_accumulator_oracle_and_changed_input_graph():
    torch.manual_seed(41)
    # The active producer-prefill regime; small shapes use the model's baseline.
    hidden = torch.randn(8192, 1536, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(6144, 1536, dtype=hidden.dtype, device="cuda") * 0.02
    zero_bias = torch.zeros(6144, dtype=hidden.dtype, device="cuda")

    def operation():
        return linear_gelu(hidden, weight, zero_bias=zero_bias)

    graph, captured = _capture(operation)
    selected_rows = [0, 127, 4096, 8191]
    for _ in range(2):
        hidden.copy_(torch.randn_like(hidden))
        weight.mul_(-0.5)
        actual = operation()
        # Lt applies GELU before the final BF16 cast, with no BF16 GEMM round.
        projected = F.linear(hidden[selected_rows].double(), weight.double())
        expected = F.gelu(projected, approximate="tanh").bfloat16()
        torch.testing.assert_close(
            actual[selected_rows], expected, atol=0.015625, rtol=0.01
        )
        _assert_replay(graph, captured, actual)


@pytest.mark.parametrize("width", [256, 6144])
@torch.inference_mode()
def test_gelu_mul_retains_bf16_intermediate_round_and_replays(width):
    torch.manual_seed(43)
    gate = torch.randn(65, width, dtype=torch.bfloat16, device="cuda") * 3
    up = torch.randn_like(gate)

    def operation():
        return gelu_tanh_mul(gate, up)

    graph, captured = _capture(operation)
    for _ in range(3):
        gate.copy_(torch.randn_like(gate) * 3)
        up.copy_(torch.randn_like(up))
        actual = operation()
        expected = F.gelu(gate, approximate="tanh") * up
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
        _assert_replay(graph, captured, actual)


@torch.inference_mode()
def test_reserved_csr_append_lookup_replays_positions_and_recycled_pages():
    from resin.backend.sampling import SAMPLING_OFFSET_STRIDE

    rows, width, window = 32, 4, 512
    initial = np.resize(
        [58, 63, 64, 65, 511, 512, 513, 8191, 8192, 8193, 8703, 8704, 8705], rows
    ).astype(np.int64)
    initial[-1] = 1  # Padded scratch row owns only physical parent zero.
    inputs = {
        "positions": torch.as_tensor(initial, device="cuda"),
        "sampling_offsets": torch.empty(rows, dtype=torch.int64, device="cuda"),
    }
    tables, firsts = {}, {}
    for kind, page_width in (("local", 137), ("global", 160)):
        first = (
            np.maximum(0, (initial + 7) // 64 + 1 - page_width)
            if kind == "local"
            else np.zeros(rows, dtype=np.int64)
        )
        table = np.zeros((rows, page_width), dtype=np.int32)
        for row in range(rows - 1):
            count = (initial[row] + 7) // 64 - first[row] + 1
            table[row, :count] = (np.arange(count) * 17 + row * 271 + 31) % 10007 + 1
        tables[kind], firsts[kind] = table, first
        inputs[f"envelope_{kind}_page_table"] = torch.as_tensor(table, device="cuda")
        inputs[f"envelope_{kind}_first_pages"] = torch.as_tensor(first, device="cuda")
        for prefix in ("assistant_", "verify_"):
            inputs[f"{prefix}{kind}_kv_indptr"] = torch.empty(
                rows + 1, dtype=torch.int32, device="cuda"
            )
            inputs[f"{prefix}{kind}_page_indices"] = torch.zeros(
                rows * page_width, dtype=torch.int32, device="cuda"
            )
            inputs[f"{prefix}{kind}_last_page_lengths"] = torch.empty(
                rows, dtype=torch.int32, device="cuda"
            )
        for field in ("append_pages", "append_offsets"):
            inputs[f"verify_{kind}_{field}"] = torch.empty(
                rows * width, dtype=torch.int64, device="cuda"
            )

    def operation():
        return update_reserved_paged_metadata(
            inputs, query_width=width, sliding_window=window
        )

    graph, _ = _capture(operation)
    for step in (0, 1, 4, 2):
        positions = initial + (np.arange(rows) + step) % 5
        positions[-1] = 1
        inputs["positions"].copy_(torch.as_tensor(positions, device="cuda"))
        graph.replay()
        np.testing.assert_array_equal(
            inputs["sampling_offsets"].cpu().numpy(),
            (positions + 1) * SAMPLING_OFFSET_STRIDE,
        )
        for kind in ("local", "global"):
            for prefix, end in (
                ("assistant_", positions),
                ("verify_", positions + width),
            ):
                first = (
                    np.maximum(0, positions - window) // 64
                    if prefix == "assistant_" and kind == "local"
                    else firsts[kind]
                )
                counts = (end + 63) // 64 - first
                np.testing.assert_array_equal(
                    inputs[f"{prefix}{kind}_kv_indptr"].cpu().numpy(),
                    np.concatenate(([0], np.cumsum(counts))),
                )
                expected = np.concatenate(
                    [
                        tables[kind][
                            row,
                            int(first[row] - firsts[kind][row]) : int(
                                first[row] - firsts[kind][row] + counts[row]
                            ),
                        ]
                        for row in range(rows)
                    ]
                )
                np.testing.assert_array_equal(
                    inputs[f"{prefix}{kind}_page_indices"][: len(expected)]
                    .cpu()
                    .numpy(),
                    expected,
                )
                np.testing.assert_array_equal(
                    inputs[f"{prefix}{kind}_last_page_lengths"].cpu().numpy(),
                    (end - 1) % 64 + 1,
                )
            absolute = positions[:, None] + np.arange(width)[None]
            expected = np.take_along_axis(
                tables[kind], absolute // 64 - firsts[kind][:, None], axis=1
            )
            np.testing.assert_array_equal(
                inputs[f"verify_{kind}_append_pages"]
                .cpu()
                .numpy()
                .reshape(rows, width),
                expected,
            )
            np.testing.assert_array_equal(
                inputs[f"verify_{kind}_append_offsets"]
                .cpu()
                .numpy()
                .reshape(rows, width),
                absolute % 64,
            )
    assert inputs["verify_local_last_page_lengths"][-1].item() == 5


@pytest.fixture
def _reference_precision(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def _attention_oracle(query, keys, values, lengths, indices, *, width, window):
    outputs, logsumexp = [], []
    for row, length in enumerate(lengths):
        key = keys[indices[row].long()].flatten(0, 1)[:length, 0].float()
        value = values[indices[row].long()].flatten(0, 1)[:length, 0].float()
        q = query[row * width : (row + 1) * width].float()
        scores = torch.einsum("qhd,kd->qhk", q, key)
        positions = torch.arange(length, device=query.device)
        query_positions = length - width + torch.arange(width, device=query.device)
        valid = positions[None, :] <= query_positions[:, None]
        if window >= 0:
            valid &= positions[None, :] >= query_positions[:, None] - window
        scores.masked_fill_(~valid[:, None, :], -torch.inf)
        outputs.append(torch.einsum("qhk,kd->qhd", scores.softmax(-1), value))
        # FA2 returns the base-2 accumulator, not natural-log logsumexp.
        logsumexp.append(scores.logsumexp(-1) / math.log(2))
    return torch.cat(outputs), torch.cat(logsumexp)


def _metadata(lengths, *, page_width=132, variant=0):
    counts = [(length + 63) // 64 for length in lengths]
    indptr = torch.zeros(len(counts) + 1, dtype=torch.int32)
    indptr[1:] = torch.tensor(counts, dtype=torch.int32).cumsum(0)
    # Distinct row banks and recycled physical order expose stale page IDs.
    rows = [
        (row * page_width + (torch.arange(count) + 17 * variant) % page_width)
        for row, count in enumerate(counts)
    ]
    indices = torch.cat(rows).to(device="cuda", dtype=torch.int32)
    last = torch.tensor(
        [(length - 1) % 64 + 1 for length in lengths], dtype=torch.int32
    )
    return indptr, indices, last, rows


def _plan(wrapper, metadata, *, heads, dimension, width, window, uniform=False):
    indptr, indices, last, _ = metadata
    kwargs = (
        {"q_len_per_req": width, "is_causal": width > 1}
        if uniform
        else {"causal": True}
    )
    args = (indptr, indices, last)
    if not uniform:
        args = (torch.arange(0, len(indptr) * width, width, dtype=torch.int32), *args)
    wrapper.plan(
        *args,
        heads,
        1,
        dimension,
        64,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=1.0,
        window_left=window,
        disable_split_kv=True,
        **kwargs,
    )


@pytest.mark.parametrize("dimension,window", [(256, 511), (512, -1)])
@pytest.mark.parametrize("batch", [1, 8])
@torch.inference_mode()
def test_uniform_q4_verify_matches_causal_oracle_and_changed_input_graph(
    dimension, window, batch, _reference_precision
):
    from resin_models_cuda.gemma4.edge_kernels import UniformCausalPagedVerifyWrapper

    torch.manual_seed(119)
    capacity = batch * 132
    keys = (
        torch.randn(capacity, 64, 1, dimension, device="cuda", dtype=torch.bfloat16)
        * 0.2
    )
    values = torch.randn_like(keys)
    query = (
        torch.randn(batch * 4, 8, dimension, device="cuda", dtype=torch.bfloat16) * 0.2
    )
    wrapper = UniformCausalPagedVerifyWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        query_width=4,
        split_kv=True,
        use_cuda_graph=True,
        backend="fa2",
        paged_kv_indptr_buf=torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
        paged_kv_indices_buf=torch.empty(capacity, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len_buf=torch.empty(batch, dtype=torch.int32, device="cuda"),
    )
    _plan(
        wrapper,
        _metadata([65] * batch),
        heads=8,
        dimension=dimension,
        width=4,
        window=window,
    )
    graph, captured = _capture(lambda: wrapper.run(query, (keys, values)))
    for iteration, length in enumerate(
        [65, 66, 127, 128, 129, 511, 512, 513, 8192, 8193]
    ):
        lengths = [length + row % 3 for row in range(batch)]
        metadata = _metadata(lengths, variant=iteration % 2)
        _plan(wrapper, metadata, heads=8, dimension=dimension, width=4, window=window)
        query.add_(0.001)
        expected, expected_lse = _attention_oracle(
            query, keys, values, lengths, metadata[3], width=4, window=window
        )
        eager, lse = wrapper.run(query, (keys, values), return_lse=True)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(eager.float(), expected, rtol=0.015, atol=0.003)
        torch.testing.assert_close(lse.float(), expected_lse, rtol=2e-4, atol=3e-4)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def _schedule(wrapper, batch, width, heads):
    info = tuple(wrapper._plan_info)
    assert len(info) == 15 and info[13] and not info[14], info
    padded, _, total_rows, tile, request, qo, kv, _, out, chunk, _, _, valid, _, _ = (
        info
    )
    real_ctas = batch * math.ceil(width * heads / tile)
    ranges = [
        (request, real_ctas * 4),
        (qo, real_ctas * 4),
        (kv, real_ctas * 4),
        (out, (batch + 1) * 4),
        (chunk, 4),
        (valid, padded),
        (total_rows, 4),
    ]
    # Ignore uninitialized padded request/tile entries; block_valid masks them.
    integer = wrapper._int_workspace_buffer
    return info, tuple(
        bytes(integer[start : start + count].cpu().tolist()) for start, count in ranges
    )


@pytest.mark.parametrize(
    "width,heads,dimension,window",
    [
        (1, 4, 256, 511),
        (1, 4, 512, -1),
        (4, 8, 256, 511),
        (4, 8, 512, -1),
    ],
)
@pytest.mark.parametrize("batch", [32, 256])
@torch.inference_mode()
def test_no_split_schedule_is_invariant_and_graph_reads_changed_csr(
    batch, width, heads, dimension, window, _reference_precision
):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    torch.manual_seed(781 + dimension + width)
    keys = (
        torch.randn(batch * 132, 64, 1, dimension, device="cuda", dtype=torch.bfloat16)
        * 0.25
    )
    values = torch.randn_like(keys) * 0.25
    query = (
        torch.randn(
            batch * width, heads, dimension, device="cuda", dtype=torch.bfloat16
        )
        * 0.25
    )
    indptr_buffer = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
    indices_buffer = torch.empty(batch * 132, dtype=torch.int32, device="cuda")
    last_buffer = torch.empty(batch, dtype=torch.int32, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buffer,
        paged_kv_indices_buffer=indices_buffer,
        paged_kv_last_page_len_buffer=last_buffer,
    )
    choices = (1, 8, 9, 127, 128, 129, 130)
    cases = []
    for variant in range(4):
        lengths = []
        for row in range(batch):
            count = choices[(row + variant) % len(choices)]
            last = (width, 63, 64, 1)[(row + variant) % 4]
            lengths.append(
                (count - 1) * 64 + max(last, width)
                if count == 1
                else (count - 1) * 64 + last
            )
        cases.append((lengths, _metadata(lengths, variant=variant)))
    schedule = None
    for _, metadata in cases:
        _plan(
            wrapper,
            metadata,
            heads=heads,
            dimension=dimension,
            width=width,
            window=window,
            uniform=True,
        )
        current = _schedule(wrapper, batch, width, heads)
        if schedule is None:
            schedule = current
        assert current == schedule, "fixed no-split schedule changed with runtime CSR"
    _plan(
        wrapper,
        cases[0][1],
        heads=heads,
        dimension=dimension,
        width=width,
        window=window,
        uniform=True,
    )
    graph, captured = _capture(lambda: wrapper.run(query, (keys, values)))
    frozen_plan = wrapper._plan_info
    for lengths, metadata in cases:
        indptr, indices, last, page_rows = metadata
        # Deliberately no plan() call after capture. Only mutable CSR and Q change.
        indptr_buffer.copy_(indptr)
        indices_buffer[: len(indices)].copy_(indices)
        last_buffer.copy_(last)
        query.copy_(torch.randn_like(query) * 0.25)
        graph.replay()
        expected, _ = _attention_oracle(
            query, keys, values, lengths, page_rows, width=width, window=window
        )
        torch.cuda.synchronize()
        assert wrapper._plan_info is frozen_plan
        torch.testing.assert_close(captured.float(), expected, rtol=0.02, atol=0.002)


def _bf16_ulp_bound(reference):
    # One BF16 rounding of the true value is at most half an ulp; allow one ulp.
    return reference.abs().clamp_min(2.0**-126) * 2.0**-7


@pytest.mark.parametrize("rows,columns", [(1536, 2048), (1536, 4096), (256, 1536), (1536, 256)])
@torch.inference_mode()
def test_m1_linear_matches_fp64_oracle_and_replays_changed_inputs(rows, columns):
    torch.manual_seed(rows + columns)
    layer = torch.nn.Linear(columns, rows, bias=False, dtype=torch.bfloat16, device="cuda")
    layer.weight.copy_(torch.randn_like(layer.weight) * 0.05)
    x = torch.randn(1, columns, dtype=torch.bfloat16, device="cuda")

    def operation():
        return linear_m1(layer, x)

    graph, captured = _capture(operation)
    for scale in (1.0, 30.0, 0.001):
        x.copy_(torch.randn_like(x) * scale)
        actual = operation()
        oracle = (x.double() @ layer.weight.double().T)
        assert torch.all((actual.double() - oracle).abs() <= _bf16_ulp_bound(oracle) + 1e-30)
        baseline_error = (layer(x).double() - oracle).abs().max()
        assert (actual.double() - oracle).abs().max() <= baseline_error * 1.01 + 1e-30
        _assert_replay(graph, captured, actual)
    # Cancellation: exact opposite weights must not leak a stale or nonzero sum.
    x.fill_(1)
    layer.weight[:, : columns // 2] = 0.25
    layer.weight[:, columns // 2 :] = -0.25
    assert linear_m1(layer, x).abs().max() == 0


@pytest.mark.parametrize("width", [6144, 12288])
@torch.inference_mode()
def test_m1_gate_up_gelu_matches_fp64_oracle_and_replays(width):
    torch.manual_seed(width)
    gate = torch.nn.Linear(1536, width, bias=False, dtype=torch.bfloat16, device="cuda")
    up = torch.nn.Linear(1536, width, bias=False, dtype=torch.bfloat16, device="cuda")
    for layer in (gate, up):
        layer.weight.copy_(torch.randn_like(layer.weight) * 0.05)
    x = torch.randn(1, 1536, dtype=torch.bfloat16, device="cuda")

    def operation():
        fused = gate_up_gelu_m1(gate, up, x)
        assert fused is not None
        return fused

    graph, captured = _capture(operation)
    for scale in (1.0, 4.0):
        x.copy_(torch.randn_like(x) * scale)
        actual = operation()
        g = (x.double() @ gate.weight.double().T)
        u = (x.double() @ up.weight.double().T)
        oracle = 0.5 * g * (1 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g**3))) * u
        existing = gelu_tanh_mul(gate(x), up(x))
        assert torch.isfinite(actual).all()
        assert (actual.double() - oracle).abs().max() <= (existing.double() - oracle).abs().max() * 1.5 + 1e-3
        _assert_replay(graph, captured, actual)


@torch.inference_mode()
def test_m1_helpers_fall_back_outside_their_contract():
    layer = torch.nn.Linear(1536, 1536, bias=False, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(2, 1536, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(linear_m1(layer, x), layer(x), atol=0, rtol=0)
    gate = torch.nn.Linear(1536, 6144, bias=False, dtype=torch.bfloat16, device="cuda")
    assert gate_up_gelu_m1(gate, gate, x) is None
    assert gate_up_gelu_m1(gate, gate, x[:1].float()) is None


def _paged_oracle(query, keys, values, pages, length, window):
    width = query.shape[-1]
    flat_keys = keys[pages.long()].reshape(-1, width)[:length].double()
    flat_values = values[pages.long()].reshape(-1, width)[:length].double()
    first = max(0, length - 1 - window) if window >= 0 else 0
    scores = query.double() @ flat_keys[first:].T
    return torch.softmax(scores, dim=-1) @ flat_values[first:]


def _paged_batch(lengths, width, *, page=64, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = sum((n + page - 1) // page for n in lengths) + 5
    keys = torch.randn(total, page, 1, width, dtype=torch.bfloat16, device="cuda", generator=generator)
    values = torch.randn_like(keys)
    pool = torch.randperm(total, device="cuda", generator=generator).to(torch.int32)
    spans, cursor = [], 0
    for n in lengths:
        count = (n + page - 1) // page
        spans.append(pool[cursor : cursor + count])
        cursor += count
    indptr = torch.tensor([0, *torch.tensor([len(x) for x in spans]).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    indices = torch.cat(spans).contiguous()
    last = torch.tensor([n - (len(x) - 1) * page for n, x in zip(lengths, spans)], dtype=torch.int32, device="cuda")
    query = torch.randn(len(lengths), 8, width, dtype=torch.bfloat16, device="cuda", generator=generator)
    return keys, values, query, indptr, indices, last, spans


@pytest.mark.parametrize("width,window", [(256, 511), (256, -1), (512, -1), (512, 127)])
@torch.inference_mode()
def test_paged_decode_matches_fp64_oracle(width, window):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    lengths = [1, 63, 64, 65, 300, 511, 512, 513, 1500]
    keys, values, query, indptr, indices, last, spans = _paged_batch(lengths, width, seed=width + window)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2", use_tensor_cores=True)
    wrapper.plan(indptr, indices, last, 8, 1, width, 64, window_left=window,
                 q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, o_data_type=torch.bfloat16,
                 sm_scale=1.0, pos_encoding_mode="NONE")
    reference = wrapper.run(query, (keys, values))
    actual = paged_decode_q1(query, keys, values, indptr, indices, last, window_left=window, page_size=64)
    assert actual.shape == query.shape and torch.isfinite(actual).all()
    for row, (length, pages) in enumerate(zip(lengths, spans)):
        oracle = _paged_oracle(query[row], keys, values, pages, length, window)
        mine = (actual[row].double() - oracle).abs().max()
        flashinfer = (reference[row].double() - oracle).abs().max()
        assert mine <= flashinfer * 1.25 + 1e-3, (length, mine.item(), flashinfer.item())
        assert mine <= 0.03


@torch.inference_mode()
def test_paged_decode_graph_reads_changed_lengths_and_tables_and_pads_safely():
    width, window = 256, 511
    keys, values, query, indptr, indices, last, spans = _paged_batch([200, 700], width, seed=3)
    capacity = torch.zeros(64, dtype=torch.int32, device="cuda")
    capacity[: len(indices)] = indices
    indptr_buffer = torch.cat((indptr, indptr[-1:])).contiguous()  # third request has zero pages
    last_buffer = torch.cat((last, last[-1:] * 0)).contiguous()
    query3 = torch.cat((query, query[:1])).contiguous()

    def operation():
        return paged_decode_q1(query3, keys, values, indptr_buffer, capacity, last_buffer, window_left=window, page_size=64)

    graph, captured = _capture(operation)
    assert torch.count_nonzero(captured[2]) == 0  # padded empty row stays zero
    for lengths, seed in (([64, 129], 5), ([450, 33], 6)):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        order = torch.randperm(keys.shape[0], device="cuda", generator=generator).to(torch.int32)
        counts = [(n + 63) // 64 for n in lengths]
        new_spans = [order[: counts[0]], order[counts[0] : counts[0] + counts[1]]]
        capacity.zero_()
        capacity[: sum(counts)] = torch.cat(new_spans)
        indptr_buffer.copy_(torch.tensor([0, counts[0], sum(counts), sum(counts)], dtype=torch.int32))
        last_buffer.copy_(torch.tensor([lengths[0] - (counts[0] - 1) * 64, lengths[1] - (counts[1] - 1) * 64, 0], dtype=torch.int32))
        query3.copy_(torch.randn_like(query3))
        graph.replay()
        torch.cuda.synchronize()
        for row in range(2):
            oracle = _paged_oracle(query3[row], keys, values, new_spans[row], lengths[row], window)
            assert (captured[row].double() - oracle).abs().max() <= 0.03
        assert torch.count_nonzero(captured[2]) == 0


@torch.inference_mode()
def test_paged_decode_rejects_unsupported_inputs():
    keys, values, query, indptr, indices, last, _ = _paged_batch([100], 256)
    with pytest.raises(ValueError):
        paged_decode_q1(query.float(), keys, values, indptr, indices, last, window_left=-1, page_size=64)
    with pytest.raises(ValueError):
        paged_decode_q1(query, keys, values, indptr.long(), indices, last, window_left=-1, page_size=64)
    with pytest.raises(ValueError):
        paged_decode_q1(query[..., :128].contiguous(), keys[..., :128], values[..., :128], indptr, indices, last, window_left=-1, page_size=64)


@pytest.mark.parametrize("width,tokens", [(256, 1), (256, 77), (512, 1024)])
@torch.inference_mode()
def test_kv_append_matches_indexed_assignment_on_permuted_arena_and_replays(width, tokens):
    pages, page = 40, 64
    storage = torch.randn(pages, 2, 6, page, 1, width, dtype=torch.bfloat16, device="cuda")
    view = storage.permute(1, 2, 0, 3, 4, 5)  # same layout as the runner's arena
    expected_storage = storage.clone()
    expected = expected_storage.permute(1, 2, 0, 3, 4, 5)
    generator = torch.Generator(device="cuda").manual_seed(width + tokens)
    slots = torch.randperm(pages * page, device="cuda", generator=generator)[:tokens]
    page_ids, offsets = (slots // page), (slots % page)
    key = torch.randn(tokens, 1, width, dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    wide = torch.randn(tokens, 3 * width, dtype=torch.bfloat16, device="cuda")
    value = wide[:, width : 2 * width].reshape(tokens, 1, width)  # strided producer view
    value.copy_(torch.randn_like(value))
    expected[0, 2][page_ids, offsets] = key
    expected[1, 2][page_ids, offsets] = value
    assert kv_append(view[0, 2], view[1, 2], key, value, page_ids, offsets)
    torch.testing.assert_close(storage, expected_storage, atol=0, rtol=0)

    def operation():
        assert kv_append(view[0, 3], view[1, 3], key, value, page_ids, offsets)
        return view[0, 3]

    graph, _ = _capture(operation)
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    expected[0, 3][page_ids, offsets] = key
    expected[1, 3][page_ids, offsets] = value
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(storage, expected_storage, atol=0, rtol=0)


@torch.inference_mode()
def test_kv_append_declines_unsupported_contracts_without_writing():
    keys = torch.zeros(4, 64, 1, 256, dtype=torch.bfloat16, device="cuda")
    values = torch.zeros_like(keys)
    key = torch.ones(2, 1, 256, dtype=torch.bfloat16, device="cuda")
    pages = torch.tensor([1, 2], device="cuda")
    offsets = torch.tensor([3, 4], device="cuda")
    assert not kv_append(keys, values, key.float(), key.float(), pages, offsets)
    assert not kv_append(keys, values, key, key, pages.float(), offsets)
    assert not kv_append(keys, values, key[:1], key, pages, offsets)
    assert torch.count_nonzero(keys) == 0 and torch.count_nonzero(values) == 0


@torch.inference_mode()
def test_fused_assistant_step_matches_reference_step_with_flashinfer_decode():
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    from resin.models.gemma4.config import FULL_ATTENTION, SLIDING_ATTENTION
    from resin_models_cuda.gemma4.edge_assistant import (
        Gemma4EdgeAssistantConfig,
        Gemma4EdgeAssistantForCausalLM,
    )

    config = Gemma4EdgeAssistantConfig(
        backbone_hidden_size=1536, vocab_size=4096, hidden_size=256, intermediate_size=2048,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
        num_global_key_value_heads=1, head_dim=256, global_head_dim=512, sliding_window=512,
        max_position_embeddings=4096, rms_norm_eps=1e-6,
        layer_types=(SLIDING_ATTENTION,) * 3 + (FULL_ATTENTION,),
        local_rope_theta=10000.0, global_rope_theta=1000000.0, global_partial_rotary_factor=0.25,
        num_centroids=64, centroid_intermediate_top_k=8,
    )
    torch.manual_seed(11)
    model = Gemma4EdgeAssistantForCausalLM(config).to("cuda")
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.04)
    for layer in model.model.layers:
        for norm in (layer.input_layernorm, layer.post_attention_layernorm,
                     layer.pre_feedforward_layernorm, layer.post_feedforward_layernorm,
                     layer.self_attn.q_norm):
            norm.weight.data.fill_(1.0)
        layer.layer_scalar.fill_(0.7)
    model.model.norm.weight.data.fill_(1.0)
    model.masked_embedding.token_ordering.copy_(torch.randperm(4096))
    model.materialize_buffers(torch.device("cuda"))
    model.eval()
    page, length = 64, 300
    pages = 8
    local_k = torch.randn(pages, page, 1, 256, dtype=torch.bfloat16, device="cuda")
    local_v = torch.randn_like(local_k)
    global_k = torch.randn(pages, page, 1, 512, dtype=torch.bfloat16, device="cuda")
    global_v = torch.randn_like(global_k)
    count = (length + page - 1) // page
    indptr = torch.tensor([0, count], dtype=torch.int32, device="cuda")
    indices = torch.arange(count, dtype=torch.int32, device="cuda")
    last = torch.tensor([length - (count - 1) * page], dtype=torch.int32, device="cuda")

    class Plan:
        kv_indptr, page_indices, last_page_lengths = indptr, indices, last

    def wrapper(window, width):
        workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        decode = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2", use_tensor_cores=False)
        decode.plan(indptr, indices, last, 4, 1, width, page, window_left=window,
                    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, o_data_type=torch.bfloat16,
                    sm_scale=1.0, pos_encoding_mode="NONE")
        return decode

    local_wrapper, global_wrapper = wrapper(511, 256), wrapper(-1, 512)
    embeds = torch.randn(1, 2 * 1536, dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([length], dtype=torch.int64, device="cuda")
    common = dict(local_wrapper=local_wrapper, global_wrapper=global_wrapper,
                  local_kv=(local_k, local_v), global_kv=(global_k, global_v))
    expected_logits, expected_hidden = Gemma4EdgeAssistantForCausalLM.__mro__[1].forward_step(
        model, embeds, positions, **common)
    actual_logits, actual_hidden = model.forward_step(
        embeds, positions, **common, local_plan=Plan, global_plan=Plan, window_left=511, page_size=page)
    scale = expected_hidden.float().abs().max()
    assert torch.isfinite(actual_hidden).all() and torch.isfinite(actual_logits).all()
    assert (actual_hidden.float() - expected_hidden.float()).abs().max() <= 0.06 * scale + 0.02
    top = expected_logits.float().topk(3).indices
    assert actual_logits.float().argmax() in top[0]
    # Without plans the original code path runs unchanged.
    fallback_logits, fallback_hidden = model.forward_step(embeds, positions, **common)
    torch.testing.assert_close(fallback_hidden, expected_hidden, atol=0, rtol=0)
