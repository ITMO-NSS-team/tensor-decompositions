import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from examples.experiment_utils import causal_lm_perplexity, find_projection_scheduler, tensor_storage_bytes, memory_metrics
from examples.mlflow_metrics_pandas_converter import MlflowMetricsPandasConverter
from examples.metric_visualizer import MetricVisualizer
from examples.tensorgrad_train import run_comparison
from tdecomp.grad_proj.tensorgrad import ULTG, AdamW


class FixedLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        probabilities = torch.tensor([[[0.25, 0.5, 0.25], [0.1, 0.2, 0.7], [0.1, 0.8, 0.1],
                                       [0.5, 0.25, 0.25], [0.3, 0.3, 0.4]]], dtype=torch.float64)
        self.register_buffer("logits", probabilities.log())
        self.fail = False

    def forward(self, **kwargs):
        if self.fail:
            raise RuntimeError("model failed")
        return {"logits": self.logits}


def language_batch():
    return {"input_ids": torch.zeros(1, 5, dtype=torch.long),
            "labels": torch.tensor([[0, 1, -100, 1, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 1]])}


@pytest.mark.parametrize("training", [True, False])
def test_perplexity_masks_labels_and_restores_model_mode(training):
    model, batch = FixedLanguageModel(), language_batch()
    model.train(training)
    before = {name: value.clone() for name, value in batch.items()}
    assert causal_lm_perplexity(model, [batch]) == pytest.approx(math.sqrt(8))
    assert model.training is training
    for name, value in batch.items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_perplexity_empty_and_failure_restore_mode():
    model, batch = FixedLanguageModel(), language_batch()
    assert causal_lm_perplexity(model, []) is None
    batch["labels"].fill_(-100)
    assert causal_lm_perplexity(model, [batch]) is None
    model.fail = True
    with pytest.raises(RuntimeError, match="model failed"):
        causal_lm_perplexity(model, [language_batch()])
    assert model.training
    with pytest.raises(ValueError):
        causal_lm_perplexity(model, [], max_batches=0)


def test_perplexity_restores_mixed_submodule_modes():
    model = FixedLanguageModel()
    model.dropout = torch.nn.Dropout().eval()
    assert model.training and not model.dropout.training
    causal_lm_perplexity(model, [language_batch()])
    assert model.training and not model.dropout.training
    model.fail = True
    with pytest.raises(RuntimeError):
        causal_lm_perplexity(model, [language_batch()])
    assert model.training and not model.dropout.training


def test_scheduler_lookup_handles_one_group_wrappers_and_adamw():
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer, _ = ULTG(model, "truncated_svd", (1, 0.5))
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    scheduler = optimizer.state[model.weight]["first_proj"].update_gap_scheduler
    assert len(optimizer.param_groups) == 1
    assert find_projection_scheduler(SimpleNamespace(optimizer=optimizer)) is scheduler
    baseline, _ = AdamW(model)
    assert find_projection_scheduler(baseline) is None
    assert find_projection_scheduler(None) is None
    metrics = memory_metrics(model, optimizer)
    moment_bytes = sum(t.untyped_storage().nbytes() for name, t in optimizer.state[model.weight].items()
                       if torch.is_tensor(t) and "exp_avg" in name)
    assert metrics["mem_opt_state_mb"] * 1024**2 > moment_bytes


def test_tensor_storage_counts_projector_data_and_deduplicates_views():
    shared = torch.ones(6, dtype=torch.float64)
    class Projector:
        _data_fields = ("basis",)
        def __init__(self):
            self.basis = torch.ones(2, 2, dtype=torch.float32)
    projector = Projector()
    assert tensor_storage_bytes({"moment": shared, "view": shared[:2], "projector": projector}) == (64, 0)


class Page(list):
    def __init__(self, values, token=None):
        super().__init__(values)
        self.token = token


class RecordingClient:
    def __init__(self):
        self.calls = []
        self.runs = {name: SimpleNamespace(info=SimpleNamespace(run_id=name, run_name="run-" + name),
                    data=SimpleNamespace(metrics={"loss": 5.0, "memory": 12.0}, params={"rank": "2"}))
                     for name in ("a", "b")}

    def get_run(self, run_id):
        return self.runs[run_id]

    def get_metric_history(self, run_id, metric):
        values = [(1, 30, 5.0), (0, 10, 8.0), (1, 20, 6.0)] if metric == "loss" else [(0, 10, 10.0), (1, 20, 11.0), (1, 30, 12.0)]
        return [SimpleNamespace(step=step, timestamp=timestamp, value=value) for step, timestamp, value in values]

    def search_runs(self, experiment_ids, **options):
        self.calls.append((experiment_ids, options))
        return Page([self.runs["b"]]) if options.get("page_token") else Page([self.runs["a"]], token="next")


def test_metric_histories_deduplicate_steps_and_align_by_main_metric():
    converter = MlflowMetricsPandasConverter(RecordingClient())
    frame = converter.get_metrics_from_run("a", include_parameters=["rank"])
    assert frame["step"].tolist() == [0, 1]
    assert frame["loss"].tolist() == [8.0, 5.0]
    assert frame["memory"].tolist() == [10.0, 12.0]
    assert frame["rank"].tolist() == ["2", "2"]
    with pytest.raises(ValueError, match="missing run parameters"):
        converter.get_metrics_from_run("a", include_parameters=["missing"])
    with pytest.raises(ValueError, match="collide"):
        converter.get_metrics_from_run("a", include_parameters=["loss"])


def test_experiment_conversion_paginates_and_resets_indices():
    client = RecordingClient()
    converter = MlflowMetricsPandasConverter(client)
    frame = converter.get_metrics_from_all_experiment_runs(7, include_parameters=["rank"])
    assert frame.index.tolist() == [0, 1, 2, 3]
    assert frame["run_id"].tolist() == ["a", "a", "b", "b"]
    assert client.calls[0][0] == ["7"]
    assert client.calls[1][1]["page_token"] == "next"
    last = converter.get_only_last_metrics_from_all_experiment_runs(7)
    assert last["run_id"].tolist() == ["a", "b"]


def test_plotting_uses_positional_legend_and_validates_columns():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    frame = pd.DataFrame({"step": [1, 0], "loss": [5, 8], "run_id": ["a", "a"],
                          "run_name": ["sample", "sample"]}, index=[20, 10])
    original = frame.copy(deep=True)
    fig, ax = MetricVisualizer.plot_metric_by_run(frame, show=False)
    assert ax.lines[0].get_label() == "sample"
    assert list(ax.lines[0].get_xdata()) == [0, 1]
    pd.testing.assert_frame_equal(frame, original)
    plt.close(fig)
    with pytest.raises(ValueError, match="run_name"):
        MetricVisualizer.plot_metric_by_run(frame.drop(columns=["run_name"]), show=False)


def test_cpu_training_comparison_is_seeded_and_uses_identical_starts():
    before = torch.random.get_rng_state()
    first, second = run_comparison(steps=2), run_comparison(steps=2)
    assert [row["optimizer"] for row in first["records"]] == ["AdamW", "ParallelTG", "ULTG"]
    assert len({row["initial_weights_sha256"] for row in first["records"]}) == 1
    for left, right in zip(first["records"], second["records"]):
        assert left["status"] == "ok"
        assert left["losses"] == right["losses"]
        assert left["evaluation_perplexity"] == right["evaluation_perplexity"]
        assert math.isfinite(left["evaluation_perplexity"])
        assert left["mem_opt_state_mb"] > 0
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


class RecordingLogger:
    def __init__(self, active=True):
        self.active, self.records = active, []

    def active_run(self):
        return self if self.active else None

    def log_metrics(self, values, step):
        self.records.append((dict(values), step))


def test_optional_native_trainer_callbacks_handle_cpu_baseline_and_inactive_runs(monkeypatch):
    pytest.importorskip("transformers", reason="optional training adapter; core measurements tested separately")
    from transformers import TrainerCallback
    from examples import experiment_utils as helpers
    logger = RecordingLogger()
    state = SimpleNamespace(global_step=5)
    model = torch.nn.Linear(4, 3, bias=False)
    optimizer, _ = ULTG(model, "truncated_svd", (1, 0.5))
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    callback = helpers.UpdateGapMLflowCallback(logger=logger)
    assert isinstance(callback, TrainerCallback)
    callback.on_step_end(None, state, None, optimizer=SimpleNamespace(optimizer=optimizer))
    assert logger.records[-1][0]["ug_next_update"] == optimizer.state[model.weight]["first_proj"].update_gap_scheduler.next_update
    baseline, _ = AdamW(model)
    before = len(logger.records)
    callback.on_step_end(None, state, None, optimizer=baseline)
    assert len(logger.records) == before
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU callback must not query CUDA allocations")
    monkeypatch.setattr(torch.cuda, "memory_allocated", forbidden)
    monkeypatch.setattr(torch.cuda, "memory_reserved", forbidden)
    helpers.SystemMetricsCallback(logger=logger).on_step_end(None, state, None, model=model)
    assert set(logger.records[-1][0]) == {"cpu_percent"}
    helpers.PreciseMemoryCallback(logger=logger).on_pre_optimizer_step(None, state, None, model=model, optimizer=optimizer)
    assert logger.records[-1][0]["mem_opt_state_mb"] > 0
    logger.active = False
    before = len(logger.records)
    helpers.SystemMetricsCallback(logger=logger).on_step_end(None, state, None, model=model)
    helpers.PreciseMemoryCallback(logger=logger).on_pre_optimizer_step(None, state, None, model=model, optimizer=optimizer)
    assert len(logger.records) == before


def test_optional_perplexity_callback_ignores_empty_evaluation_and_inactive_runs():
    pytest.importorskip("transformers", reason="optional training adapter; core measurements tested separately")
    from examples.experiment_utils import PerplexityCallback
    model, logger = FixedLanguageModel(), RecordingLogger()
    state = SimpleNamespace(global_step=5)
    callback = PerplexityCallback(logger=logger)
    callback.on_step_end(None, state, None, model=model, eval_dataloader=[language_batch()])
    assert logger.records[-1] == ({"perplexity": pytest.approx(math.sqrt(8))}, 5)
    before = len(logger.records)
    callback.on_step_end(None, state, None, model=model, eval_dataloader=[], train_dataloader=[language_batch()])
    assert len(logger.records) == before
    logger.active, model.fail = False, True
    callback.on_step_end(None, state, None, model=model, eval_dataloader=[language_batch()])
    assert len(logger.records) == before
    with pytest.raises(ValueError):
        PerplexityCallback(eval_every_n_steps=0)


def test_galore_profiler_ranges_preserve_reconstruction():
    from tdecomp.grad_proj.tensorgrad.projectors.galore_projector import GaLoreProjector
    source = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    projector = GaLoreProjector(2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        compressed = projector.project(source, 0)
        reconstructed = projector.project_back(compressed)
    keys = {event.key for event in profiler.key_averages()}
    assert {"tdecomp.galore.project", "tdecomp.galore.project_back"} <= keys
    expected = projector.ortho_matrix @ (projector.ortho_matrix.mT @ source)
    torch.testing.assert_close(reconstructed, expected)
