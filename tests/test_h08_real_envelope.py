"""Real-tail rank feasibility and energy certificate independent of training."""
import importlib.util
from pathlib import Path
import pytest
import torch
import tensorly as tl

pytest.importorskip('torchvision')
path=Path(__file__).parents[1]/'experiments/hypotheses/run_h08_real.py'
spec=importlib.util.spec_from_file_location('h08_real_envelope',path)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_real_tail_candidates_do_not_exceed_structural_rank():
    for batch in (128,80,56,16):
        ranks=runner.real_rank_candidates(batch)
        assert len(ranks)==len(set(ranks))
        assert all(a<=batch and c<=min(256,4*batch) and (h,w)==(2,2) for a,c,h,w in ranks)
    assert runner.real_rank_candidates(16)==((8,32,2,2),(16,64,2,2))
    with pytest.raises(ValueError):runner.real_rank_candidates(17)


@pytest.mark.parametrize('epsilon',[.10,.5])
def test_energy_direct_residual_matches_certificate_and_is_scale_covariant(epsilon):
    x=torch.randn(9,7,2,2,generator=torch.Generator().manual_seed(788),dtype=torch.float64)
    with tl.backend_context('pytorch'):
        estimate,info=runner.energy_reconstruction(x,epsilon)
        scaled,scaled_info=runner.energy_reconstruction(11*x,epsilon)
    assert info['relative_error']<=epsilon+1e-10
    assert info['ranks']==scaled_info['ranks']
    torch.testing.assert_close(scaled,11*estimate,rtol=1e-9,atol=1e-9)
    # Orthogonal projection: discarded energy equals the direct residual energy.
    torch.testing.assert_close((x-estimate).square().sum(),x.square().sum()-estimate.square().sum(),rtol=1e-9,atol=1e-9)


def test_clean_activation_intervention_is_exact_native_forward():
    from torchvision.models import resnet18
    model=resnet18(weights=None,num_classes=10).eval()
    images=torch.randn(2,3,32,32,generator=torch.Generator().manual_seed(105))
    with torch.inference_mode():
        reference=model(images)
        clean=runner.prefix(model,images)
        intervened=runner.suffix(model,clean)
    assert clean.shape==(2,256,2,2)
    torch.testing.assert_close(intervened,reference,rtol=0,atol=0)
