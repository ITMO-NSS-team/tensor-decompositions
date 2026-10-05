import itertools
import math

import pytest
import torch

from experiments.hypotheses import run_h10_posthoc_feasibility as audit


@pytest.fixture(scope="module",autouse=True)
def bounded_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_all2520_labeled_assignments_match_independent_full_permutation_oracle():
    placements=audit.balanced_placements()
    reference=set()
    for permutation in itertools.permutations(range(8)):
        mapping=[0]*8
        for slot,expert in enumerate(permutation):
            mapping[expert]=slot//2
        reference.add(tuple(mapping))
    assert len(reference)==math.factorial(8)//(2**4)==2520
    assert set(map(tuple,placements.tolist()))==reference
    assert all(row.count(device)==2 for row in placements.tolist() for device in range(4))


def test_loads_match_independent_python_sums_and_exact_threshold():
    expert=torch.tensor([12,7,6,4,3,2,1,0],dtype=torch.int64)
    placements=audit.balanced_placements()
    actual=audit.placement_loads(expert,placements)
    expected=torch.tensor([[sum(int(expert[e]) for e in range(8) if placement[e]==device) for device in range(4)]
                           for placement in placements.tolist()])
    assert torch.equal(actual,expected)
    assert torch.equal(16*actual.max(1).values<=5*int(expert.sum()),
                       actual.max(1).values.double()<=1.25*float(expert.sum())/4)


def test_single_expert_necessary_bound_is_not_sufficient_for_pair_capacity():
    expert=torch.tensor([8,8,8,8,8,0,0,0],dtype=torch.int64)
    assert audit.necessary_bound(expert)["necessary_max_expert_bound_passed"]
    rows,_,_=audit.layer_feasibility(dict.fromkeys(audit.SPLITS,expert),audit.balanced_placements())
    assert all(row["feasible_assignments"]==0 for row in rows)
    assert rows[0]["minimum_achievable_worst_device_over_split_average"]==pytest.approx(1.6)
    assert rows[0]["reason"]=="infeasible_by_exhaustive_pair_capacity_and_split_constraints"


def test_impossible_heavy_expert_is_rejected_before_any_claim_about_greedy():
    expert=torch.tensor([20,1,1,1,1,1,1,1],dtype=torch.int64)
    assert not audit.necessary_bound(expert)["necessary_max_expert_bound_passed"]
    rows,_,_=audit.layer_feasibility(dict.fromkeys(audit.SPLITS,expert),audit.balanced_placements())
    assert all(row["feasible_assignments"]==0 for row in rows)
    assert rows[0]["reason"]=="infeasible_already_by_single_expert_necessary_bound"


def test_joint_splits_are_checked_separately_not_pooled():
    cal=torch.tensor([10,10,10,10,0,0,0,0],dtype=torch.int64)
    tune=10-cal
    rows,_,_=audit.layer_feasibility({"calibration":cal,"tuning":tune,"future":cal},audit.balanced_placements())
    assert 0<rows[1]["feasible_assignments"]<2520
    merged=cal+tune
    assert bool((16*audit.placement_loads(merged,audit.balanced_placements()).max(1).values<=5*int(merged.sum())).all())
    assert rows[1]["splits_checked_separately"]==["calibration","tuning"]


def test_future_oracle_can_rule_out_every_joint_placement_without_changing_primary():
    patterns=[(0,0,0),(0,0,0),(0,1,1),(0,1,1),(1,0,1),(1,0,1),(1,1,0),(1,1,0)]
    vectors={split:torch.tensor([10*p[k] for p in patterns],dtype=torch.int64) for k,split in enumerate(audit.SPLITS)}
    rows,individual,_=audit.layer_feasibility(vectors,audit.balanced_placements())
    assert all(bool(mask.any()) for mask in individual.values())
    assert rows[1]["feasible_assignments"]>0
    assert rows[2]["feasible_assignments"]==0
    assert rows[2]["future_used_as_retrospective_oracle"]
    assert all(row["new_primary_placement_authorized"] is False for row in rows)


def test_rejected_stored_placement_does_not_establish_infeasibility():
    expert=torch.tensor([10,10,0,0,10,10,0,0],dtype=torch.int64)
    stored=torch.tensor([[0,1,2,3,0,1,2,3]])
    assert int(audit.placement_loads(expert,stored).max())==20
    rows,_,_=audit.layer_feasibility(dict.fromkeys(audit.SPLITS,expert),audit.balanced_placements())
    assert rows[0]["feasible_assignments"]>0
    witness=torch.tensor([rows[0]["first_feasible_assignment_diagnostic_only"]])
    assert int(audit.placement_loads(expert,witness).max())==10


def test_zero_and_integer_boundary_have_no_floating_slack():
    zero=torch.zeros(8,dtype=torch.int64)
    assert audit.necessary_bound(zero)["max_expert_over_average_device_load"] is None
    rows,_,_=audit.layer_feasibility(dict.fromkeys(audit.SPLITS,zero),audit.balanced_placements())
    assert all(row["feasible_assignments"]==2520 for row in rows)
    boundary=torch.tensor([5,0,4,0,4,0,3,0],dtype=torch.int64)
    assert audit.necessary_bound(boundary)["maximum_allowed_device_load"]==5
    assert audit.necessary_bound(boundary)["necessary_max_expert_bound_passed"]
    over=torch.tensor([6,0,4,0,4,0,2,0],dtype=torch.int64)
    assert not audit.necessary_bound(over)["necessary_max_expert_bound_passed"]


def test_completed_count_validation_rejects_nonintegral_partial_and_invalid_origin():
    tensor=torch.zeros(2,4,8,12)
    tensor[0,0,0,0]=3
    assert int(audit.integer_counts(tensor,2,3).sum())==3
    with pytest.raises(ValueError,match="completed CPU"):
        audit.integer_counts(tensor,3,3)
    with pytest.raises(ValueError,match="does not match"):
        audit.integer_counts(tensor,2,4)
    tensor[0,0,0,0]=3.5
    with pytest.raises(ArithmeticError,match="integral"):
        audit.integer_counts(tensor,2,3)
    tensor[0,0,0,0]=3
    tensor[0,1,0,0]=1
    with pytest.raises(ValueError,match="multiple original origin"):
        audit.integer_counts(tensor,2,4)


def test_core_never_queries_cuda(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("posthoc feasibility must use CPU only")
    for name in ("is_available","current_device","mem_get_info","synchronize","reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    placements=audit.balanced_placements()
    load=torch.ones(8,dtype=torch.int64)
    rows,_,_=audit.layer_feasibility(dict.fromkeys(audit.SPLITS,load),placements)
    assert all(row["feasible_assignments"]==2520 for row in rows)
