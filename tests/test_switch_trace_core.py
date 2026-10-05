import torch
import pytest
pytest.importorskip('transformers')
from transformers import SwitchTransformersConfig,SwitchTransformersForConditionalGeneration
from experiments.hypotheses.switch_trace_core import (
    corrupt,AcceptedTrace,physical_permutation,sparse_modules,greedy_placement,
)


def tiny_switch():
    torch.manual_seed(17)
    config=SwitchTransformersConfig(vocab_size=64,d_model=16,d_ff=32,num_layers=2,
        num_decoder_layers=2,num_heads=2,d_kv=8,num_experts=4,num_sparse_encoder_layers=1,
        num_sparse_decoder_layers=1,expert_capacity=2,dropout_rate=0,decoder_start_token_id=0,
        pad_token_id=0,eos_token_id=1,router_jitter_noise=0)
    return SwitchTransformersForConditionalGeneration(config).eval()


def test_storage_permutation_preserves_ties_capacity_and_restores_after_error():
    model=tiny_switch()
    for _,module in sparse_modules(model):
        with torch.no_grad():module.router.classifier.weight.zero_()
    ids=torch.tensor([[3,4,5,6,7,8]]);labels=torch.tensor([[9,10,1]])
    payload={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}
    trace=AcceptedTrace(model);trace.begin(0,payload)
    with torch.no_grad():reference=model(**payload).logits
    original=trace.finish();masks={k:v.clone() for k,v in trace.route_masks.items()}
    before={k:v.clone() for k,v in model.state_dict().items()}
    placements=[[1,0,1,0] for _ in sparse_modules(model)]
    with pytest.raises(ValueError):
        with physical_permutation(model,placements):
            trace.begin(0,payload)
            with torch.no_grad():actual=model(**payload).logits
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            torch.testing.assert_close(trace.finish(),original)
            for index in masks:torch.testing.assert_close(trace.route_masks[index],masks[index])
            raise ValueError('simulate interrupted replay')
    for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert all(int(original[0,index])==2 for index in range(original.shape[1]))
    trace.close()


def test_corruption_has_exact_noise_budget_and_decoder_sentinel_sequence():
    source=list(range(10,138));sentinels=list(range(999,900,-1))
    payload,info=corrupt(source,123,sentinels,1,0)
    assert info['masked_tokens']==19 and info['spans']==6
    labels=payload['labels'][0].tolist()
    assert [value for value in labels if value>=900]==sentinels[:7]
    assert sum(value in source for value in labels)==19
    assert int(payload['attention_mask'].sum())==128-19+6+1
    assert labels[-1]==1 and len(labels)<=64
    again,_=corrupt(source,123,sentinels,1,0)
    for key in payload:torch.testing.assert_close(payload[key],again[key])


def test_greedy_filters_load_before_selecting_cheapest_device():
    counts=torch.zeros(1,2,4,1)
    counts[0,0,:,0]=torch.tensor([4,3,2,1])
    placement,details=greedy_placement(counts,'cost',devices=2,capacity=2)
    assert details['admissible']
    assert placement==[[0,1,0,1]]
    assert details['layers'][0]['device_load']==[6,4]
