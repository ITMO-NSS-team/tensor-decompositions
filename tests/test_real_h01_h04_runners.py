"""CPU-only independent operator and common-BN admission for real runners."""
import copy

import pytest
import torch
pytest.importorskip('torchvision')
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from experiments.hypotheses import run_h01_real as h01
from experiments.hypotheses import run_h04_real as h04


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def test_direct_tucker_matches_dense_for_output_input_and_factor_gradients():
    rows = h04.admission()
    assert [row["dtype"] for row in rows] == ["torch.float64", "torch.float32"]
    assert rows[0]["forward_relative"] < 1e-10
    assert rows[0]["gradient_relative_max"] < 1e-10
    assert rows[1]["forward_relative"] < 1e-5
    assert rows[1]["gradient_relative_max"] < 1e-5


def test_common_bn_updates_only_affected_buffers_and_full_rank_convolution_is_no_intervention():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(987)
        teacher = models.resnet18(weights=None, num_classes=10).double().eval()
    gen = torch.Generator().manual_seed(843)
    x = torch.randn(4, 3, 32, 32, generator=gen, dtype=torch.float64)
    loader = DataLoader(TensorDataset(x, torch.zeros(4, dtype=torch.long)), batch_size=2)
    before = copy.deepcopy(teacher.state_dict())
    h01.calibrate_bn(teacher, loader, "cpu")
    affected = "layer3.0.bn2."
    changed = [key for key, value in teacher.state_dict().items() if not torch.equal(value, before[key])]
    assert changed
    assert set(changed).issubset({affected + name for name in ("running_mean", "running_var", "num_batches_tracked")})
    assert int(teacher.layer3[0].bn2.num_batches_tracked) == 2
    assert all(not module.training for module in teacher.modules() if isinstance(module, nn.BatchNorm2d))
    student = copy.deepcopy(teacher)
    weight = teacher.layer3[0].conv2.weight.detach()
    # Full-rank A=I and B=W test the concrete 256-channel MatrixConv padding
    # and all copied BN/residual operations independently of SVD accuracy.
    student.layer3[0].conv2 = h04.MatrixConv(torch.eye(256, dtype=torch.float64), weight.reshape(256, -1))
    student.eval()
    with torch.no_grad():
        torch.testing.assert_close(student(x), teacher(x), atol=1e-10, rtol=1e-10)
    for name, module in student.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            other = teacher.get_submodule(name)
            torch.testing.assert_close(module.running_mean, other.running_mean, atol=0, rtol=0)
            torch.testing.assert_close(module.running_var, other.running_var, atol=0, rtol=0)


def test_support_solution_matches_independent_weighted_diagonal_oracle():
    w = torch.diag(torch.tensor([10., 1., 2.], dtype=torch.float64))
    moment = torch.diag(torch.tensor([1e-4, 1., 0.], dtype=torch.float64))
    a, b, info = h01.support_solution(w, moment, 1)
    approximation = a @ b
    assert info["ridge"] == 0
    assert info["support_dimension"] == 2
    torch.testing.assert_close(approximation, torch.diag(torch.tensor([0., 1., 0.], dtype=torch.float64)),
                               atol=1e-12, rtol=1e-12)
    risk = torch.einsum("ij,jk,ik->", w - approximation, moment, w - approximation)
    assert float(risk) == pytest.approx(0.01, abs=1e-12)
