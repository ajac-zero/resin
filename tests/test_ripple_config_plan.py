from __future__ import annotations

import math
from dataclasses import replace

import pytest
from output_helpers import forward_output

from resin import (
    BatchPhase,
    BatchPlan,
    CacheRetention,
    Engine,
    RunnerConfig,
    SamplingParams,
    SchedulerConfig,
)
from resin.backend.cuda.parallel import TensorParallelLayout
from resin.models.ripple.config import (
    FULL_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    RippleConfig,
    attention_geometry,
)
from resin.models.ripple.plan import build_ripple_batch_plan, ripple_cache_layout
from ripple.tiny_config import real_config_dict, tiny_config_dict


@pytest.fixture(scope="module")
def real() -> RippleConfig:
    return RippleConfig.from_dict(real_config_dict())


@pytest.fixture(scope="module")
def tiny() -> RippleConfig:
    return RippleConfig.from_dict(tiny_config_dict())


class TestRippleConfig:
    def test_real_config(self, real: RippleConfig) -> None:
        assert real.num_hidden_layers == 80
        assert (real.local_layer_count, real.global_layer_count) == (64, 16)
        assert real.history_blocks_per_parent == 4
        assert real.layer_types[:6] == ("sliding_attention",) * 4 + (
            "full_attention",
            "sliding_attention",
        )
        assert real.embed_scale == pytest.approx(math.sqrt(5120))
        assert real.ffn_clamp_threshold == 20.0
        assert real.eos_token_ids == (100265, 100257) and real.pad_token_id == 100277
        assert real.use_valres and real.use_gated_attention and real.gate_full_precision
        assert not real.tie_word_embeddings

    def test_tiny_config(self, tiny: RippleConfig) -> None:
        assert (tiny.local_layer_count, tiny.global_layer_count) == (8, 2)
        assert tiny.history_blocks_per_parent == 4
        assert tiny.embed_scale == pytest.approx(math.sqrt(tiny.hidden_size))

    def test_layer_cache_slots_number_layers_within_each_group(
        self, tiny: RippleConfig
    ) -> None:
        slots = tiny.layer_cache_slots
        assert slots[:6] == ((1, 0), (1, 1), (1, 2), (1, 3), (0, 0), (1, 4))
        assert slots[9] == (0, 1)
        assert [tiny.layer_cache_group(i) for i in (3, 4)] == [1, 0]

    def test_same_shape_reduction_keeps_cache_packing(self, real: RippleConfig) -> None:
        reduced = real.same_shape_reduced()
        assert reduced.layer_types == ("sliding_attention",) * 4 + ("full_attention",)
        assert (reduced.local_layer_count, reduced.global_layer_count) == (4, 1)
        assert reduced.history_blocks_per_parent == 4
        assert reduced.hidden_size == real.hidden_size
        assert real.same_shape_reduced(10).history_blocks_per_parent == 4
        with pytest.raises(ValueError, match="history_blocks_per_parent"):
            real.same_shape_reduced(6)  # S S S S G S: 5 local per global
        with pytest.raises(ValueError, match="num_layers"):
            real.same_shape_reduced(81)

    def test_default_layer_types_follow_the_hybrid_period(self) -> None:
        raw = tiny_config_dict()
        del raw["layer_types"]
        assert (
            RippleConfig.from_dict(raw).layer_types
            == (("sliding_attention",) * 4 + ("full_attention",)) * 2
        )

    def test_zero_clamp_disables_clamping_like_the_reference(self) -> None:
        raw = tiny_config_dict() | {"ffn_clamp_threshold": 0, "embed_scale": None}
        config = RippleConfig.from_dict(raw)
        assert config.ffn_clamp_threshold is None and config.embed_scale is None

    @pytest.mark.parametrize(
        "change,message",
        [
            ({"model_type": "llama"}, "expected a Ripple checkpoint"),
            ({"num_key_value_heads": 3}, "divisible by KV heads"),
            ({"head_dim": 15}, "head_dim must be even"),
            ({"rms_norm_eps": 0}, "rms_norm_eps must be positive"),
            ({"sliding_window": 0}, "must be positive"),
            ({"tie_word_embeddings": True}, "untied"),
            ({"ffn_clamp_threshold": -1.0}, "ffn_clamp_threshold"),
            ({"layer_types": ["linear_attention"] * 10}, "unsupported Ripple layer"),
            ({"layer_types": ["full_attention"] * 9}, "one item per hidden layer"),
            ({"eos_token_id": []}, "eos_token_ids must not be empty"),
            ({"pad_token_id": 256}, "pad_token_id"),
            ({"hidden_act": "gelu"}, "silu"),
            (
                {"use_gated_attention": False, "gate_full_precision": True},
                "requires use_gated_attention",
            ),
        ],
    )
    def test_invalid_configs_are_rejected(self, change, message) -> None:
        with pytest.raises(ValueError, match=message):
            RippleConfig.from_dict(tiny_config_dict() | change)

    def test_cache_packing_requires_both_layer_kinds_and_an_integral_ratio(
        self, tiny: RippleConfig
    ) -> None:
        all_local = replace(tiny, layer_types=("sliding_attention",) * 10)
        with pytest.raises(ValueError, match="both sliding and full"):
            all_local.history_blocks_per_parent
        uneven = replace(
            tiny, layer_types=("sliding_attention",) * 5 + ("full_attention",) * 5
        )
        assert uneven.history_blocks_per_parent == 1
        odd = replace(
            tiny, layer_types=("sliding_attention",) * 7 + ("full_attention",) * 3
        )
        with pytest.raises(ValueError, match="integral parent"):
            odd.history_blocks_per_parent


class TestTensorParallelGeometry:
    @pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
    def test_real_ripple_shards_cleanly(self, real: RippleConfig, size: int) -> None:
        geometries = [
            attention_geometry(real, TensorParallelLayout(rank, size))
            for rank in range(size)
        ]
        assert {g.q_heads for g in geometries} == {80 // size}
        assert {g.intermediate_size for g in geometries} == {40960 // size}
        assert [g.q_head_start for g in geometries] == [
            rank * (80 // size) for rank in range(size)
        ]
        assert {g.kv_heads for g in geometries} == {max(1, 8 // size)}
        assert {g.kv_replicas for g in geometries} == {max(1, size // 8)}

    @pytest.mark.parametrize("size", [1, 2, 4])
    def test_tiny_ripple_shards_cleanly(self, tiny: RippleConfig, size: int) -> None:
        geometries = [
            attention_geometry(tiny, TensorParallelLayout(rank, size))
            for rank in range(size)
        ]
        assert sum(g.q_heads for g in geometries) == 20
        assert sum(g.intermediate_size for g in geometries) == 1280
        # Every query shard reads the one KV head that holds its group.
        group = tiny.num_attention_heads // tiny.num_key_value_heads
        for g in geometries:
            first, last = g.q_head_start, g.q_head_start + g.q_heads - 1
            if g.kv_heads == 1:
                assert first // group == last // group == g.kv_head_start

    @pytest.mark.parametrize(
        "size,message",
        [(8, "num_attention_heads=20"), (3, "divisible by TP=3")],
    )
    def test_unsupported_tiny_layouts_are_rejected(
        self, tiny: RippleConfig, size, message
    ) -> None:
        with pytest.raises(ValueError, match=message):
            attention_geometry(tiny, TensorParallelLayout(0, size))

    def test_kv_heads_that_do_not_divide_are_rejected(self, real: RippleConfig) -> None:
        with pytest.raises(ValueError, match="cannot shard across TP=5"):
            attention_geometry(real, TensorParallelLayout(0, 5))
        six_kv = replace(real, num_attention_heads=60, num_key_value_heads=6)
        with pytest.raises(ValueError, match="cannot shard across TP=4"):
            attention_geometry(six_kv, TensorParallelLayout(0, 4))


class TestRippleCachePlan:
    BLOCK = 4
    CHUNK = 8

    def _layout(self, config: RippleConfig, capacity: int = 64):
        return ripple_cache_layout(
            config,
            parent_capacity=capacity,
            block_tokens=self.BLOCK,
            prefill_chunk_size=self.CHUNK,
        )

    def test_layout_groups(self, tiny: RippleConfig) -> None:
        layout = self._layout(tiny)
        full, window = (
            layout.groups[FULL_CACHE_GROUP],
            layout.groups[WINDOW_CACHE_GROUP],
        )
        assert full.blocks_per_slab == tiny.history_blocks_per_parent == 4
        assert full.retention == CacheRetention.HISTORY and full.window_tokens is None
        assert window.retention == CacheRetention.WINDOW
        assert window.read_window_tokens == tiny.sliding_window
        assert window.window_tokens == self.CHUNK + tiny.sliding_window - 1
        with pytest.raises(ValueError, match="prefill_chunk_size"):
            ripple_cache_layout(
                tiny, parent_capacity=4, block_tokens=4, prefill_chunk_size=0
            )

    def test_engine_plans_translate_with_window_eviction(
        self, tiny: RippleConfig
    ) -> None:
        layout = self._layout(tiny)
        translated = []

        class Runner:
            runner_config = RunnerConfig(
                continuation_prefill=True, cache_layout=layout, prefill_chunk_size=8
            )

            @staticmethod
            def bind_resources(_binding) -> None:
                return None

            def forward(self, plan: BatchPlan):
                translated.append((plan, build_ripple_batch_plan(plan, layout)))
                return forward_output([7] * plan.num_sequences)

        engine = Engine(
            Runner(), scheduler_config=SchedulerConfig(prefill_chunk_size=8)
        )
        prompt = list(range(1, 22))
        request = engine.submit(
            prompt, sampling_params=SamplingParams(max_output_tokens=8, temperature=0)
        )
        while engine.has_work():
            engine.step()
        assert request.take_result() == [7] * 8

        prefill = [p for _, p in translated if p.phase == BatchPhase.PREFILL]
        decode = [p for _, p in translated if p.phase == BatchPhase.DECODE]
        assert [len(p.token_ids) for p in prefill] == [8, 8, 5]
        assert len(decode) == 7
        for _, plan in translated:
            position = int(plan.positions[-1])
            first = plan.window.first_token_positions[0]
            assert plan.full.first_token_positions == (0,)
            assert plan.full.kv_lengths == (position + 1,)
            assert plan.window.kv_lengths == (position + 1 - first,)
            # The window group keeps every key a query can read and only whole blocks.
            assert first % self.BLOCK == 0
            assert first <= max(0, int(plan.positions[0]) - (tiny.sliding_window - 1))
            assert plan.full.last_page_lengths[0] == (position % self.BLOCK) + 1
            assert plan.window.last_page_lengths[0] == (position % self.BLOCK) + 1
            # One native append slot per scheduled token, in each group.
            assert len(plan.full.append_slots) == len(plan.window.append_slots)
            assert len(plan.full.append_slots) == len(plan.token_ids)
        assert decode[-1].window.first_token_positions[0] > 0
        assert decode[-1].window.kv_lengths[0] < decode[-1].full.kv_lengths[0]
        assert engine.cache_free_parent_count == 64

    def test_rejects_layouts_without_two_groups(self, tiny: RippleConfig) -> None:
        from resin import CacheGroupLayout, CacheLayout

        layout = CacheLayout(parent_capacity=2, groups=[CacheGroupLayout(2)])
        plan = BatchPlan(
            sequence_ids=[1], context_lengths=[0], token_ids=[1], token_indptr=[0, 1]
        )
        with pytest.raises(ValueError, match="full and a window group"):
            build_ripple_batch_plan(plan, layout)
