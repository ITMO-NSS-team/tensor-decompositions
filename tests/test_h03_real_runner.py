"""Independent checks of global native-GPT2 algebra and real data boundaries."""
import copy
from types import SimpleNamespace

import pytest
import torch
pytest.importorskip('transformers')
pytest.importorskip('pyarrow')
from transformers import GPT2Config, GPT2LMHeadModel

from experiments.hypotheses import gpt2_rotation_core as core
from experiments.hypotheses import run_h03_real as runner


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_model(dtype=torch.float64):
    torch.manual_seed(57)
    config = GPT2Config(n_embd=32, n_head=4, n_layer=3, n_positions=16, n_ctx=16,
                        vocab_size=37, resid_pdrop=0, attn_pdrop=0, embd_pdrop=0)
    return core.disable_dropout(GPT2LMHeadModel(config).to(dtype=dtype))


def test_native_gpt2_all_blocks_and_final_head_compensated_fp64_fp32():
    results = core.small_admission()
    assert len(results) == 10
    assert max(result["precompression_output_error"] for name, result in results.items() if "float64" in name) < 1e-10
    assert max(result["precompression_output_error"] for name, result in results.items() if "float32" in name) < 1e-5


def test_width768_exact_identity_lift_and_nonzero_zero_cayley_derivative():
    width = 768
    reference = torch.zeros(1, width)
    e = core.complement(width, reference)
    eye = torch.eye(width - 1)
    assert torch.equal(core.lift(eye, e), torch.eye(width))
    assert torch.equal(core.choose_rotation("identity", reference, 101, e), torch.eye(width))
    assert torch.equal(core.choose_rotation("cayley", reference, 101, e), torch.eye(width))
    # One independently specified skew coefficient at the origin.
    coordinate = torch.tensor(0., requires_grad=True)
    direction = torch.zeros_like(eye)
    direction[0, 1], direction[1, 0] = 1, -1
    a = direction * coordinate
    p = torch.linalg.solve((eye + a).T, (eye - a).T).T
    q = core.lift(p, e)
    assert torch.equal(q, torch.eye(width))
    derivative, = torch.autograd.grad(q[0, 1], coordinate)
    expected = (-2 * (e[:, 0, None] @ e[None, :, 1]) + 2 * (e[:, 1, None] @ e[None, :, 0]))[0, 1]
    torch.testing.assert_close(derivative, expected, atol=1e-6, rtol=1e-5)
    assert abs(float(derivative)) > .01


def test_analytic_householder_lift_matches_dense_formula_and_width768_mean_constraint():
    for width, dtype in ((32, torch.float64), (768, torch.float32)):
        hidden = torch.randn(16, width, generator=torch.Generator().manual_seed(46), dtype=dtype)
        e = core.complement(width, hidden)
        for method in ("hadamard", "haar"):
            q = core.choose_rotation(method, hidden, 101, e)
            assert float((q @ torch.ones(width, dtype=dtype) - 1).norm()) / width**.5 < 1e-5
            assert float((q.double() @ torch.ones(width, dtype=torch.float64) - 1).norm()) / width**.5 < 1e-6
            assert float((q.T @ q - torch.eye(width, dtype=dtype)).norm()) / width < 1e-6
        if dtype == torch.float64:
            p = core.orthogonal_qr_cleanup(torch.randn(width-1, width-1, generator=torch.Generator().manual_seed(94), dtype=dtype))
            direct = e @ p @ e.T + torch.ones(width, width, dtype=dtype)/width
            torch.testing.assert_close(core.lift(p, e), direct, atol=1e-12, rtol=1e-12)


def test_explicit_fp64_assembly_casts_back_to_fp32_and_keeps_cayley_gradient():
    base = core.untie_and_absorb(small_model(torch.float32))
    block = core.CalibrationBlock(base.transformer.h[1], torch.eye(32), cayley=True,
                                  assembly_dtype=torch.float64)
    q = block.rotation()
    assert q.dtype == torch.float32
    assert torch.equal(q, torch.eye(32))
    q[0, 1].backward()
    assert block.skew_coordinates.grad.dtype == torch.float32
    assert block.skew_coordinates.grad.abs().sum() > 0


def test_global_rotation_preserves_input_gradients_and_missing_position_compensation_fails():
    base = core.untie_and_absorb(small_model())
    hidden = torch.randn(64, 32, generator=torch.Generator().manual_seed(93), dtype=torch.float64)
    q = core.choose_rotation("haar", hidden, 101, core.complement(32, hidden))
    rotated = core.rotate_global_in_place(copy.deepcopy(base), q)
    inputs = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(72), dtype=torch.float64, requires_grad=True)
    a = base(inputs_embeds=inputs, use_cache=False).logits
    b = rotated(inputs_embeds=inputs @ q.T, use_cache=False).logits
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    ga, = torch.autograd.grad(a.square().sum(), inputs)
    gb, = torch.autograd.grad(b.square().sum(), inputs)
    torch.testing.assert_close(ga, gb, rtol=1e-10, atol=1e-10)
    broken = copy.deepcopy(rotated)
    broken.transformer.wpe.weight.data.copy_(base.transformer.wpe.weight)
    tokens = torch.randint(37, (2, 16), generator=torch.Generator().manual_seed(87))
    assert float((core.logits(broken, tokens) - core.logits(base, tokens)).norm()) > .01


def test_calibration_only_same_four_matrices_and_biases_plus_cayley_with_equal_batches():
    base = core.untie_and_absorb(small_model(torch.float32))
    tokens = torch.randint(37, (8, 16), generator=torch.Generator().manual_seed(99))
    hidden, targets = core.cache_block_inputs(base, tokens, "cpu", block_index=1, batch_size=2)
    fixed = core.CalibrationBlock(base.transformer.h[1], torch.eye(32))
    learned = core.CalibrationBlock(base.transformer.h[1], torch.eye(32), cayley=True)
    norm_count = sum(parameter.numel() for module in learned.modules() if isinstance(module, torch.nn.LayerNorm)
                     for parameter in module.parameters())
    assert norm_count == 0
    assert learned.skew_coordinates.numel() == 31 * 30 // 2
    assert sum(p.numel() for p in learned.block.parameters()) == sum(p.numel() for p in fixed.block.parameters())
    histories = [runner.calibrate(block, hidden, targets, 101, "cpu", steps=2, microbatch=4)
                 for block in (fixed, learned)]
    assert histories[0]["batches_sha256"] == histories[1]["batches_sha256"]
    assert all(torch.isfinite(p).all() for p in learned.parameters())
    q = learned.rotation()
    torch.testing.assert_close(q.T @ q, torch.eye(32), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q @ torch.ones(32), torch.ones(32), rtol=1e-5, atol=1e-5)


def test_conv1d_quantizes_output_columns_using_independent_integer_reference_and_ste():
    conv = base_conv = small_model().transformer.h[0].attn.c_attn
    quantized = core.QuantizedConv1D(conv).eval()
    with torch.no_grad():
        quantized.weight[:, 0].zero_()
    weight = quantized.weight.detach()
    scales = weight.abs().amax(0) / 7
    reference = weight.clone()
    for channel in range(weight.shape[1]):
        if scales[channel] == 0:
            reference[:, channel] = 0
        else:
            reference[:, channel] = (weight[:, channel] / scales[channel]).round().clamp(-7, 7) * scales[channel]
    x = torch.randn(2, 4, 32, dtype=torch.float64)
    torch.testing.assert_close(quantized(x), x @ reference + conv.bias, rtol=1e-12, atol=1e-12)
    quantized.train()
    quantized(x).square().sum().backward()
    assert quantized.weight.grad[:, 0].abs().sum() == 0
    assert quantized.weight.grad[:, 1:].abs().sum() > 0


def test_documents_preserve_every_raw_string_and_insert_eos_between_documents_only():
    rows = ["\n", " = First = \n", "Text\n", "", " = = Section = = \n", "Tail\n", " = Second = \n", "Next\n"]
    docs = runner.documents_from_rows(rows)
    assert docs == ["".join(rows[:6]), "".join(rows[6:])]
    class Tokenizer:
        eos_token_id = 999
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return [ord(character) for character in text]
    tokens, ends = runner.tokenize_rows(rows, Tokenizer())
    assert tokens.tolist() == [ord(c) for c in docs[0]] + [999] + [ord(c) for c in docs[1]]
    assert ends == [len(docs[0]) + 1, len(docs[0]) + 1 + len(docs[1])]
    windows, tail = runner.window_stream(torch.arange(131), length=16)
    assert windows.shape == (8, 16) and tail == 3
    assert windows.flatten().tolist() == list(range(128))


def test_recovery_freezes_first_tuning_checkpoint_and_never_requires_test(tmp_path):
    model = core.untie_and_absorb(small_model(torch.float32))
    windows = torch.randint(37, (8, 16), generator=torch.Generator().manual_seed(39))
    path = tmp_path / "fixed.pt"
    history, hit, _, checkpoint_hash, _ = runner.recover(model, windows, windows, 1e20, 101, "cpu", path,
                                                         steps=1, microbatch=4)
    assert hit["step"] == 0
    assert checkpoint_hash == runner.file_hash(path)
    assert history[0]["step"] == 0
    saved = torch.load(path, weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_paired_summary_marks_censored_or_incomplete_pairs_indeterminate():
    rows = []
    for seed in (101, 202, 303):
        for method, seconds in (("identity", 100.), ("haar", 100.), ("hadamard", 80.)):
            rows.append({"seed": seed, "method": method, "quality_reached": True, "total_seconds_to_quality": seconds})
    assert runner.paired_summary(rows, (101, 202, 303))[0]["ci95_low"] == pytest.approx(.2)
    rows[0]["quality_reached"] = False
    assert runner.paired_summary(rows, (101, 202, 303))[0]["outcome"] == "indeterminate"


def test_recovery_stress_pilot_has_full_model_updates_and_no_calibration_budget():
    assert runner.phase_steps(SimpleNamespace(recovery_pilot=True, pilot=True)) == (0, 20)
    assert runner.phase_steps(SimpleNamespace(recovery_pilot=False, pilot=True)) == (20, 0)
    assert runner.phase_steps(SimpleNamespace(recovery_pilot=False, pilot=False)) == (50, 200)
