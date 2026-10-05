"""Check GPU modal tail energies against an independent CPU FP32 reference."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/hypotheses"))
from run_h12_real import original_channel_spectra


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native GPU spectrum admission")
def test_cuda_modal_energy_and_tail_match_cpu_reference():
    weight = torch.randn((256, 256, 3, 3), generator=torch.Generator().manual_seed(101))
    reference = original_channel_spectra(weight)
    observed = original_channel_spectra(weight.cuda())
    assert observed['svd_driver'] == 'gesvd'
    for actual, expected in zip(observed['modes'], reference['modes']):
        assert actual['relative_energy_identity_error'] <= 1e-5
        torch.testing.assert_close(torch.tensor(actual['tail_squared']),
                                   torch.tensor(expected['tail_squared']), rtol=1e-5, atol=1e-4)
