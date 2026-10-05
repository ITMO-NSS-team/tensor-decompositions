"""Independent actual-state byte pricing and complete native Switch replay."""
import time

import pytest
import torch
pytest.importorskip('transformers')
pytest.importorskip('pyarrow')
from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration

from experiments.hypotheses import run_h10_real as h10
from experiments.hypotheses.switch_trace_core import AcceptedTrace, sparse_modules


def test_corruption_stream_ranges_are_disjoint_between_runs_and_splits():
    intervals=[]
    for seed in (101,202,303):
        for partition in ('calibration','tuning','replay','test'):
            first=h10.corruption_seed(seed,partition,0)
            last=h10.corruption_seed(seed,partition,99999)
            assert last-first==99999
            intervals.append((first,last))
    intervals.sort()
    assert all(left[1]<right[0] for left,right in zip(intervals,intervals[1:]))
    # Previously these two different examples shared the very same RNG seed.
    assert h10.corruption_seed(101,'calibration',202)!=h10.corruption_seed(202,'calibration',101)
    with pytest.raises(ValueError):h10.corruption_seed(101,'test',100000)


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_switch():
    torch.manual_seed(97)
    config = SwitchTransformersConfig(vocab_size=64, d_model=16, d_ff=32, num_layers=2,
        num_decoder_layers=2, num_heads=2, d_kv=8, num_experts=4, num_sparse_encoder_layers=1,
        num_sparse_decoder_layers=1, expert_capacity=3, dropout_rate=0, decoder_start_token_id=0,
        pad_token_id=0, eos_token_id=1, router_jitter_noise=0)
    model = SwitchTransformersForConditionalGeneration(config).eval()
    with torch.no_grad():
        for _, module in sparse_modules(model):
            module.router.classifier.weight.zero_()
    return model


class Tokenizer:
    eos_token_id, pad_token_id = 1, 0
    def convert_tokens_to_ids(self, token):
        return 63 - int(token.removeprefix("<extra_id_").removesuffix(">")) % 50


def test_ring_byte_cost_doubles_with_state_size_but_expert_migration_is_fixed():
    counts = torch.zeros(2, 4, 8, 2)
    counts[0, 0, 0, 0] = 3
    counts[0, 3, 1, 0] = 1
    counts[1, 2, 7, 1] = 2
    placements = [[e % 4 for e in range(8)] for _ in range(2)]
    placements[0][0] = 2
    expert_bytes = [[100 + e for e in range(8)], [200 + e for e in range(8)]]
    narrow = h10.price(counts, placements, expert_bytes, [2, 2])
    wide = h10.price(counts, placements, expert_bytes, [4, 4])
    expected_weighted_dispatches = 3 * 2 + 1 * 2 + 2 * 1
    assert narrow["ring_weighted_exchange_bytes"] == expected_weighted_dispatches * 2 * 768 * 2
    assert wide["ring_weighted_exchange_bytes"] == 2 * narrow["ring_weighted_exchange_bytes"]
    assert narrow["migration_bytes"] == wide["migration_bytes"] == 100
    assert wide["total_exchange_and_migration_bytes"] == 2 * narrow["ring_weighted_exchange_bytes"] + 100


def test_native_accepted_padding_and_state_dtype_are_observed_after_capacity():
    model = tiny_switch()
    chunk = {"index": 0, "tokens": [12]}
    payload, _ = h10.payload_for(chunk, "replay", 101, Tokenizer())
    trace = AcceptedTrace(model)
    try:
        trace.begin(0, payload)
        with torch.no_grad():
            model(**payload, use_cache=False)
        counts = trace.finish()
        for index, (name, _) in enumerate(trace.modules):
            assert trace.state_bytes[index] == 4
            if name.startswith("encoder."):
                assert int(counts[:, index].sum()) == 3
                assert trace.padding[index] == 1  # one accepted native padding token
                assert trace.dropped[index] == 125
            else:
                assert int(counts[:, index].sum()) == 1
                assert trace.padding[index] == 0
    finally:
        trace.close()


def test_replay_checks_every_supplied_chunk_and_preserves_exact_masks(monkeypatch):
    model = tiny_switch()
    chunks = [{"index": index, "tokens": [10, 11, 12, 13, 14, 15]} for index in range(17)]
    placements = [[1, 0, 1, 0] for _ in sparse_modules(model)]
    guard_calls = []
    # The legacy root guard probes CUDA regardless of suppliedCPU device.
    # This CPU-only admission must not initialize or query a GPU context.
    monkeypatch.setattr(h10, "guard", lambda *args: guard_calls.append(args))
    result = h10.replay_admission(model, chunks, 101, Tokenizer(), "cpu", placements, time.perf_counter())
    assert result["chunks"] == 17 and len(guard_calls) == 17
    assert result["maximum_output_error"] == 0
    assert result["route_masks_equal"] and result["accepted_and_dropped_counts_equal"]
    assert len(result["reference_routes_sha256"]) == 64
