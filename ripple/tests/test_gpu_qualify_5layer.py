"""CPU checks of the GPUBox 5-layer qualification harness.

Everything except the CUDA stages is exercised: argument parsing, the case
table, the metric and gate maths, the offline HF oracle loader (tiny two-shard
checkpoint, first-N layers only), and the Engine-driving code path against the
pure-PyTorch Ripple runner.
"""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ripple import gpu_qualify_5layer as q
from ripple.tiny_config import build_tiny_hf_model, tiny_config_dict

SCRIPT = Path(q.__file__)


def _checkpoint(directory: Path, hf, *, shard_at: int = 5) -> Path:
    state = {k: v.contiguous() for k, v in hf.state_dict().items()}
    late = {
        k: v
        for k, v in state.items()
        if k.startswith("model.layers.") and int(k.split(".")[2]) >= shard_at
    }
    early = {k: v for k, v in state.items() if k not in late}
    save_file(early, directory / "model-00001-of-00002.safetensors")
    save_file(late, directory / "model-00002-of-00002.safetensors")
    weight_map = dict.fromkeys(early, "model-00001-of-00002.safetensors")
    weight_map |= dict.fromkeys(late, "model-00002-of-00002.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (directory / "config.json").write_text(json.dumps(tiny_config_dict()))
    return directory


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ripple-tiny")
    hf = build_tiny_hf_model(seed=2)
    return _checkpoint(directory, hf), hf


class TestCli:
    def test_help_exits_zero_and_documents_the_flags(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        for flag in ("--checkpoint-dir", "--num-layers", "--output-json", "--mode"):
            assert flag in result.stdout
        assert "{all,tp1,tp2}" in result.stdout

    def test_defaults_match_the_gpubox_cache(self) -> None:
        args = q.parse_args([])
        assert args.num_layers == 5 and args.mode == "all"
        assert args.checkpoint_dir.endswith(
            "snapshots/b172323ffd63e62de9ce1ba3830dfffe493c7671"
        )
        assert q.tolerances(args) == q.Tolerances(
            min_cosine=0.9999, max_top5_logprob_err=0.05, max_abs_err=0.25
        )

    def test_invalid_arguments_are_rejected(self) -> None:
        with pytest.raises(SystemExit):
            q.parse_args(["--mode", "tp3"])
        with pytest.raises(SystemExit):
            q.parse_args(["--num-layers", "0"])

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only behaviour")
    def test_without_cuda_main_refuses_instead_of_running(self, tmp_path) -> None:
        assert q.main(["--output-json", str(tmp_path / "r.json")]) == 2

    def test_dry_run_validates_checkpoint_and_reads_only_needed_shards(
        self, tiny_dir, tmp_path
    ) -> None:
        directory, _ = tiny_dir
        output = tmp_path / "report.json"
        argv = ["--dry-run", "--checkpoint-dir", str(directory), "--output-json"]
        assert q.main([*argv, str(output), "--num-layers", "5"]) == 0
        report = json.loads(output.read_text())
        assert report["status"] == "dry-run"
        plan = report["dry_run"]
        assert plan["shards"] == ["model-00001-of-00002.safetensors"]
        assert plan["layer_types"] == ["sliding_attention"] * 4 + ["full_attention"]
        assert plan["history_blocks_per_parent"] == 4
        assert [case["name"] for case in plan["cases"]] == ["A", "B", "C"]
        assert {"timestamp_utc", "hostname", "torch", "gpus"} <= set(
            report["environment"]
        )
        # All ten layers need both shards; a missing one is an error, not a crash.
        assert q.main([*argv, str(output), "--num-layers", "10"]) == 0
        assert len(json.loads(output.read_text())["dry_run"]["shards"]) == 2
        assert q.main([*argv, str(output), "--num-layers", "11"]) == 1
        assert q.main([*argv, str(output), "--checkpoint-dir", str(tmp_path)]) == 1


class TestCases:
    def test_default_cases_match_the_qualification_plan(self) -> None:
        a, b, c = q.default_cases(1024)
        assert (a.lengths, a.steps) == ((64,), 16)
        assert b.lengths == (1023, 1024, 1025, 2100) and b.steps == 32
        assert 2100 > 2 * 1024  # window pages are evicted
        assert c.lengths == (37, 1025, 1600) and c.steps == 16 and c.batched

    def test_prompts_are_deterministic_and_in_vocabulary(self) -> None:
        first = q.make_prompt(40, 7, 100_352)
        assert first == q.make_prompt(40, 7, 100_352)
        assert first != q.make_prompt(40, 8, 100_352)
        assert all(1000 <= token < 50_000 for token in first)
        assert all(
            0 <= token < 1500 for token in q.make_prompt(40, 7, 1500)
        )  # small vocabularies clamp
        case = q.Case("B", (5, 9), 2)
        prompts = q.case_prompts(case, 256_000)
        assert [len(p) for p in prompts] == [5, 9]

    def test_graph_shapes_cover_the_longest_decode(self) -> None:
        shapes = q.graph_shapes(q.default_cases(1024), 64)
        assert [batch for batch, _ in shapes] == [1, 2, 4]
        assert all(width * 64 >= 2132 for _, width in shapes)


class TestMetrics:
    def test_identical_logits(self) -> None:
        logits = torch.randn(3, 50)
        metrics = q.compare_logits(logits, logits.clone())
        assert metrics["max_abs_err"] == 0 and metrics["max_rel_err"] == 0
        assert metrics["cosine_similarity"] == pytest.approx(1.0)
        assert metrics["top1_match_rate"] == 1.0
        assert metrics["max_top5_logprob_abs_err"] == 0

    def test_known_perturbation(self) -> None:
        expected = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]
        )
        actual = expected.clone()
        actual[0, 3] += 0.5  # |e| = 4 > floor 1, still not the argmax
        actual[1, 0] -= 3.0  # flips row 1's argmax (6 -> 3 < 5)
        metrics = q.compare_logits(actual, expected)
        assert metrics["max_abs_err"] == pytest.approx(3.0)
        assert metrics["mean_abs_err"] == pytest.approx(3.5 / 12)
        assert metrics["max_rel_err"] == pytest.approx(3.0 / 6.0)
        assert metrics["top1_match_rate"] == 0.5
        manual = max(
            (actual[r].log_softmax(-1) - expected[r].log_softmax(-1))
            .double()
            .abs()[expected[r].topk(5).indices]
            .max()
            .item()
            for r in range(2)
        )
        assert metrics["max_top5_logprob_abs_err"] == pytest.approx(manual)
        row0 = torch.nn.functional.cosine_similarity(
            actual[0].double(), expected[0].double(), dim=0
        )
        row1 = torch.nn.functional.cosine_similarity(
            actual[1].double(), expected[1].double(), dim=0
        )
        assert metrics["cosine_similarity"] == pytest.approx(float(min(row0, row1)))

    def test_shape_mismatch_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="shapes differ"):
            q.compare_logits(torch.zeros(2, 5), torch.zeros(3, 5))

    def test_trajectory_helpers(self) -> None:
        assert q.trajectory_match_length([1, 2, 3], [1, 2, 3]) == 3
        assert q.trajectory_match_length([1, 9, 3], [1, 2, 3]) == 1
        assert q.trajectory_match_length([], [1]) == 0
        # Steps through the first divergent one share a context.
        assert q.compared_steps(3, 3) == 3 and q.compared_steps(1, 3) == 2
        assert q.top2_gaps(torch.tensor([[1.0, 4.0, 3.5]])) == [0.5]

    def _generation(self, steps: int = 4, vocab: int = 40):
        generator = torch.Generator().manual_seed(0)
        logits = 3 * torch.randn(steps, vocab, generator=generator)
        return logits.argmax(-1).tolist(), logits

    def test_gates_pass_and_fail_on_each_metric(self) -> None:
        tokens, logits = self._generation()
        tolerance = q.Tolerances()
        assert q.evaluate(tokens, logits, tokens, logits, tolerance=tolerance)["passed"]
        floor = logits.argmin(-1, keepdim=True)  # far from the top: only abs error
        noisy = logits.scatter_add(1, floor, torch.full((len(tokens), 1), 0.3))
        result = q.evaluate(tokens, noisy, tokens, logits, tolerance=tolerance)
        assert not result["passed"]
        assert any("max_abs_err" in reason for reason in result["failures"])
        assert not any("top5" in reason for reason in result["failures"])
        # The cross-run budget is stricter on abs error.
        small = logits.scatter_add(1, floor, torch.full((len(tokens), 1), 0.15))
        assert q.evaluate(tokens, small, tokens, logits, tolerance=tolerance)["passed"]
        assert not q.evaluate(
            tokens, small, tokens, logits, tolerance=tolerance, cross=True
        )["passed"]

    def test_logprob_gate(self) -> None:
        tokens, logits = self._generation()
        skewed = logits.clone()
        skewed[:, tokens[0]] += 0.0
        top = logits.topk(5, dim=-1).indices
        skewed.scatter_add_(1, top[:, :1], torch.full((len(tokens), 1), 0.2))
        result = q.evaluate(tokens, skewed, tokens, logits, tolerance=q.Tolerances())
        assert result["max_top5_logprob_abs_err"] > 0.05
        assert any("top5" in reason for reason in result["failures"])

    def test_divergence_is_a_tie_only_inside_the_margin(self) -> None:
        tokens, logits = self._generation()
        tolerance = q.Tolerances(tie_margin=0.25)
        flipped = list(tokens)
        flipped[2] = (tokens[2] + 1) % 40
        top = logits[2].topk(2).values
        gap = float(top[0] - top[1])
        wide = q.evaluate(flipped, logits, tokens, logits, tolerance=tolerance)
        assert wide["trajectory_match_length"] == 2 and wide["steps_compared"] == 3
        assert wide["passed"] is (gap <= 0.25)
        close = logits.clone()
        close[2, tokens[2]] = close[2].topk(2).values[1] + 0.1  # force a near-tie
        near = q.evaluate(flipped, close, tokens, close, tolerance=tolerance)
        assert near["passed"] and near["benign_tie"]
        far = logits.clone()
        far[2, tokens[2]] = far[2].max() + 5.0
        result = q.evaluate(flipped, far, tokens, far, tolerance=tolerance)
        assert not result["passed"]
        assert any("diverged at step 2" in reason for reason in result["failures"])

    def test_memory_checks_require_half_the_sharded_weights(self) -> None:
        footprint = {
            "model_bytes": 1000,
            "layer_projection_bytes": 800,
            "kv_bytes_per_parent": 1024,
        }
        half = {**footprint, "model_bytes": 600, "kv_bytes_per_parent": 512}

        def report(rank_footprint):
            return {
                "tp1": {"eager": {"footprint": footprint}},
                "tp2": {
                    "ranks": {0: {"modes": {"eager": {"footprint": rank_footprint}}}}
                },
            }

        assert all(row["passed"] for row in q.memory_checks(report(half)))
        unsharded = {**half, "model_bytes": 1000}
        assert not all(row["passed"] for row in q.memory_checks(report(unsharded)))
        same_kv = {**half, "kv_bytes_per_parent": 1024}
        assert not all(row["passed"] for row in q.memory_checks(report(same_kv)))

    def test_summary_table_and_json_stripping(self) -> None:
        tokens, logits = self._generation()
        row = {
            "stage": "tp1_eager",
            "reference": "hf",
            "case": "A[64]",
            **q.evaluate(tokens, logits, tokens, logits, tolerance=q.Tolerances()),
        }
        table = q.format_summary([row, {**row, "case": "X", "passed": False}])
        assert "tp1_eager" in table and "PASS" in table
        stripped = q.strip_tensors({"a": {"logits": logits, "tokens": [1]}, "b": [row]})
        assert stripped["a"] == {"tokens": [1]}
        json.dumps(stripped)


class TestEnvironment:
    def test_inventory_has_every_required_field(self, monkeypatch) -> None:
        monkeypatch.setenv("NCCL_GRAPH_FILE", "/tmp/graph.xml")
        env = q.collect_environment()
        expected = {
            "timestamp_utc", "hostname", "python", "torch", "cuda_runtime",
            "cuda_driver", "nccl", "flashinfer", "triton", "transformers",
            "safetensors", "resin_git", "gpus", "NCCL_GRAPH_FILE",
        }  # fmt: skip
        assert expected <= set(env)
        assert env["NCCL_GRAPH_FILE"] == "/tmp/graph.xml"
        assert env["torch"] == torch.__version__ and env["transformers"]
        assert env["timestamp_utc"].endswith("+00:00")
        assert len(env["resin_git"]["sha"] or "") == 40
        json.dumps(env)


class TestOracle:
    def test_reduced_oracle_reads_only_first_shard_and_matches_hf(
        self, tiny_dir, tmp_path
    ) -> None:
        directory, hf = tiny_dir
        copy = tmp_path / "ckpt"
        copy.mkdir()
        for path in directory.iterdir():
            if not path.name.startswith("model-00002"):
                (copy / path.name).write_bytes(path.read_bytes())
        with pytest.raises(FileNotFoundError, match="00002"):
            q.load_hf_oracle(copy, 10, "cpu", torch.float32)
        model = q.load_hf_oracle(copy, 5, "cpu", torch.float32)
        assert model.config.num_hidden_layers == 5
        assert model.config.layer_types == ["sliding_attention"] * 4 + [
            "full_attention"
        ]
        assert not any(p.is_meta for p in model.parameters())
        ids = torch.tensor([[3, 17, 99, 4, 250, 8, 31, 2, 1, 66, 7]])
        with torch.no_grad():
            prefix = model(ids).logits
            reference = build_tiny_hf_model(seed=2)
            reference.model.layers = reference.model.layers[:5]
            reference.config.num_hidden_layers = 5
            reference.config.layer_types = reference.config.layer_types[:5]
            expected = reference(ids).logits
        torch.testing.assert_close(prefix, expected, rtol=1e-5, atol=1e-5)

    def test_greedy_decode_matches_naive_full_context_argmax(self, tiny_dir) -> None:
        directory, _ = tiny_dir
        model = q.load_hf_oracle(directory, 10, "cpu", torch.float32)
        prompt = q.make_prompt(21, 5, 250)
        tokens, logits = q.hf_greedy(model, prompt, 6, "cpu")
        context = list(prompt)
        with torch.no_grad():
            for step, token in enumerate(tokens):
                expected = model(torch.tensor([context])).logits[0, -1]
                torch.testing.assert_close(logits[step], expected, rtol=1e-4, atol=1e-4)
                assert token == int(expected.argmax())
                context.append(token)
        assert logits.shape == (6, 256)


class _ObservingSampler:
    """Greedy sampler that reports the pre-sampling logits, like the CUDA runner."""

    def __init__(self, runner) -> None:
        self.runner = runner

    def __call__(self, logits, plan):
        observer = getattr(self.runner, "logits_observer", None)
        if observer is not None:
            observer(plan, logits)
        return logits.argmax(-1)


class TestEngineDriver:
    @pytest.mark.parametrize("chunk", [4, 8])
    def test_cases_through_engine_match_the_hf_oracle(self, tiny_dir, chunk) -> None:
        from resin import Engine
        from resin.models.ripple.plan import ripple_cache_layout
        from resin_models_torch.ripple import RippleTorchRunner, load_model

        directory, _ = tiny_dir
        oracle = q.load_hf_oracle(directory, 5, "cpu", torch.float32)
        model, config = load_model(directory, device="cpu", num_layers=5)
        layout = ripple_cache_layout(
            config, parent_capacity=64, block_tokens=4, prefill_chunk_size=chunk
        )
        runner = RippleTorchRunner(model, cache_layout=layout, prefill_chunk_size=chunk)
        setattr(runner, "logits_observer", None)
        setattr(runner, "_sampler", _ObservingSampler(runner))
        cases = [
            q.Case("A", (5,), 4),
            q.Case("B", (7, 8, 9, 26), 3),
            q.Case("C", (5, 9, 21), 3, batched=True),
        ]
        args = Namespace(prefill_chunk_size=chunk)
        engine = Engine(runner, **q.engine_options(args))

        def observe(recorder):
            setattr(runner, "logits_observer", recorder)

        golden = q.build_golden(oracle, cases, 250, "cpu", lambda _message: None)
        results = q.run_cases(engine, observe, cases, 250, lambda _message: None)
        assert set(results) == set(golden) and len(results) == 8
        tolerance = q.Tolerances(max_abs_err=1e-3, max_top5_logprob_err=1e-3)
        rows = q.compare_runs(
            "cpu", results, golden, tolerance, reference_name="hf", cross=False
        )
        assert all(row["passed"] for row in rows), q.format_summary(rows)
        assert all(row["trajectory_match_length"] == row["steps"] for row in rows)
        assert all(row["steps_compared"] == row["steps"] for row in rows)
        for key, value in results.items():
            assert value["tokens"] == golden[key]["tokens"]
        engine.close()

    def test_missing_logits_row_is_detected(self) -> None:
        recorder = q.Recorder()
        recorder.rows = [(1, 4, torch.zeros(3))]  # step 0 only, two requested
        with pytest.raises(RuntimeError, match="recorded steps"):
            q.step_logits(recorder, [1], [5], 2)
