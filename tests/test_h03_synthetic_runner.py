import copy

import pytest
import torch

from experiments.hypotheses import run_h03_synthetic as h03


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_full_transformer_compensation_in_fp64_and_fp32_and_negative_controls():
    result = h03.admission()
    for dtype, tolerance in ((torch.float64, 1e-10), (torch.float32, 1e-5)):
        for method in h03.METHODS:
            assert result[f"{dtype}_{method}"]["precompression_output_error"] <= tolerance
    assert result["rope_head_commutator_positive_and_negative_controls"] == "passed"
    assert result["causal_mask_future_independence"] == "passed"


def test_quantization_matches_independent_integer_rounding_and_has_correct_ste():
    weight = torch.tensor([[1., -.43, .18, -1.], [0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    actual = h03.quant4(weight)
    independent = weight.detach().clone()
    scale = 1 / 7
    independent[0] = torch.tensor([scale * max(-7, min(7, round(float(value) / scale))) for value in weight[0]], dtype=weight.dtype)
    torch.testing.assert_close(actual, independent, rtol=0, atol=1e-15)
    ste = h03.quant4(weight, ste=True)
    ste.sum().backward()
    torch.testing.assert_close(weight.grad[0], torch.ones(4, dtype=weight.dtype))
    torch.testing.assert_close(weight.grad[1], torch.zeros(4, dtype=weight.dtype))


def test_procrustes_reduces_declared_fixed_activation_surrogate():
    teacher = h03.ToyTransformer(11, dtype=torch.float64)
    tokens = torch.randint(32, (4, 32), generator=torch.Generator().manual_seed(17))
    hidden = teacher.input_hidden(tokens).reshape(-1, 64).detach()
    complement = h03.mean_complement(dtype=torch.float64)
    initial = h03.fixed_rotation("haar", hidden, 11, complement)
    target = h03.quant4(hidden @ initial.T, channel_axis=1)
    optimized = h03.fixed_rotation("procrustes", hidden, 11, complement)
    assert float((hidden @ optimized.T - target).norm()) <= float((hidden @ initial.T - target).norm()) + 1e-12
    torch.testing.assert_close(optimized @ torch.ones(64, dtype=hidden.dtype), torch.ones(64, dtype=hidden.dtype), atol=1e-12, rtol=1e-12)


def test_cayley_zero_and_learned_skew_preserve_orthogonality_and_mean():
    teacher = h03.ToyTransformer(11, dtype=torch.float64)
    absorbed = h03.absorb_affine(teacher)
    student = h03.RotatedQuantized(absorbed, torch.eye(64, dtype=torch.float64), cayley=True)
    assert student.skew_coordinates.numel() == 1953
    with torch.no_grad():
        student.skew_coordinates.copy_(.03 * torch.randn(1953, generator=torch.Generator().manual_seed(87), dtype=torch.float64))
    q = student.rotation()
    torch.testing.assert_close(q.T @ q, torch.eye(64, dtype=q.dtype), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(q @ torch.ones(64, dtype=q.dtype), torch.ones(64, dtype=q.dtype), rtol=1e-12, atol=1e-12)


def test_fp32_qr_cleanup_removes_svd_orthogonality_loss_without_sign_changes():
    q = torch.linalg.qr(torch.randn(63, 63, generator=torch.Generator().manual_seed(607))).Q
    perturbed = q * torch.linspace(.99999, 1.00001, 63)
    repaired = h03.orthogonal_qr_cleanup(perturbed)
    torch.testing.assert_close(repaired.T @ repaired, torch.eye(63), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(repaired, q, atol=1e-6, rtol=1e-6)


def test_calibration_has_equal_batches_and_frozen_embeddings_head_positions():
    teacher = h03.ToyTransformer(11)
    absorbed = h03.absorb_affine(teacher)
    splits = h03.sequence_data(11, include_test=False)
    assert "test" not in splits
    histories = []
    for cayley in (False, True):
        student = h03.RotatedQuantized(absorbed, torch.eye(64), cayley=cayley)
        before = {name: value.clone() for name, value in student.base.state_dict().items()}
        history = h03.calibrate(student, teacher, splits["calibration"], 11, "cpu", steps=2)
        histories.append(history)
        for key in ("embedding.weight", "positions", "head.weight"):
            torch.testing.assert_close(student.base.state_dict()[key], before[key], rtol=0, atol=0)
        assert history["extra_cayley_coordinates"] == (1953 if cayley else 0)
        assert all(torch.isfinite(parameter).all() for parameter in student.parameters())
    assert histories[0]["batches_sha256"] == histories[1]["batches_sha256"]


def test_packed_nibbles_reconstruct_the_actual_quantized_weight():
    teacher = h03.ToyTransformer(11)
    student = h03.RotatedQuantized(h03.absorb_affine(teacher), torch.eye(64))
    packed = h03.packed_weights(student)
    actual = student.rotated_weights()
    for name, item in packed.items():
        bytes_ = item["packed_uint8"]
        codes = torch.stack((bytes_ & 15, bytes_ >> 4), dim=1).reshape(-1).to(torch.float32) - 7
        decoded = codes.reshape(item["shape"]) * item["scales"][:, None]
        torch.testing.assert_close(decoded, actual[name], rtol=0, atol=0)


def test_arbitrary_external_rotation_is_rejected_by_full_ln_graph_admission():
    teacher = h03.ToyTransformer(11, dtype=torch.float64)
    arbitrary = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(11), dtype=torch.float64)).Q
    student = h03.RotatedQuantized(h03.absorb_affine(teacher), arbitrary, quantized=False)
    tokens = torch.randint(32, (2, 32), generator=torch.Generator().manual_seed(77))
    with pytest.raises(ArithmeticError, match="exact rotation admission failed"):
        h03.rotation_checks(student, teacher, tokens)
