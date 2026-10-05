"""Accepted Switch dispatches, deterministic T5 corruption, and storage permutation."""
from __future__ import annotations
from contextlib import contextmanager
from types import MethodType
import torch
from torch import nn


def positive_partition(total,parts,generator):
    if not 1<=parts<=total:raise ValueError('invalid positive composition')
    cuts=sorted((torch.randperm(total-1,generator=generator)[:parts-1]+1).tolist())
    boundaries=[0,*cuts,total]
    return [b-a for a,b in zip(boundaries,boundaries[1:])]


def corrupt(tokens,seed,sentinel_ids,eos_id,pad_id):
    """Uniform positive noise/non-noise compositions; exactly rounded15% corruption."""
    n=len(tokens);generator=torch.Generator().manual_seed(seed)
    if n<2:noise=0;spans=0;mask=[False]*n
    else:
        noise=min(n-1,max(1,round(.15*n)))
        spans=min(noise,n-noise,max(1,round(noise/3)))
        clean_lengths=positive_partition(n-noise,spans,generator)
        noise_lengths=positive_partition(noise,spans,generator)
        mask=[]
        for clean,bad in zip(clean_lengths,noise_lengths):mask.extend([False]*clean+[True]*bad)
    inputs=[];labels=[];span=0;inside=False
    for token,masked in zip(tokens,mask):
        if masked:
            if not inside:
                sentinel=sentinel_ids[span];inputs.append(sentinel);labels.append(sentinel);span+=1
            labels.append(int(token));inside=True
        else:inputs.append(int(token));inside=False
    if spans:labels.append(sentinel_ids[span])
    inputs.append(eos_id);labels.append(eos_id)
    if len(inputs)>128 or len(labels)>64:raise ArithmeticError('T5 corruption exceeds protocol lengths')
    attention=[1]*len(inputs)+[0]*(128-len(inputs));inputs += [pad_id]*(128-len(inputs))
    return {'input_ids':torch.tensor(inputs)[None,:],'attention_mask':torch.tensor(attention)[None,:],
            'labels':torch.tensor(labels)[None,:]}, {'original_tokens':n,'masked_tokens':noise,'spans':spans,
                                                   'encoder_nonpadding':sum(attention),'decoder_tokens':len(labels)}


def sparse_modules(model):
    return [(name,module) for name,module in model.named_modules() if type(module).__name__=='SwitchTransformersSparseMLP']


class AcceptedTrace:
    def __init__(self,model):
        self.modules=sparse_modules(model);self.handles=[]
        for index,(name,module) in enumerate(self.modules):
            self.handles.append(module.router.register_forward_hook(self.hook(index,name)))

    def begin(self,origin,payload):
        self.origin=origin;self.payload=payload;self.values={};self.padding={};self.dropped={};self.route_masks={};self.state_bytes={}

    def hook(self,index,name):
        def capture(_,inputs,output):
            mask=output[0].detach().bool()
            self.state_bytes[index]=inputs[0].element_size()
            if bool((mask.sum(-1)>1).any()):raise ArithmeticError('Switch accepted more than one expert per token')
            self.values[index]=mask.sum((0,1)).cpu().long()
            self.dropped[index]=int((mask.sum(-1)==0).sum())
            if name.startswith('encoder.'):
                padding=~self.payload['attention_mask'].bool().to(mask.device)
            else:padding=torch.zeros(mask.shape[:2],device=mask.device,dtype=torch.bool)
            self.padding[index]=int((mask & padding[...,None]).sum())
            self.route_masks[index]=mask.cpu()
        return capture

    def finish(self):
        if len(self.values)!=len(self.modules):raise ArithmeticError('not all sparse layers were captured')
        return torch.stack([self.values[i] for i in range(len(self.modules))],dim=-1)

    def close(self):
        for handle in self.handles:handle.remove()


@contextmanager
def physical_permutation(model,placements):
    """Permute storage, retain logical router/tie/capacity decisions, then restore."""
    saved=[]
    try:
        with torch.no_grad():
            for index,(_,module) in enumerate(sparse_modules(model)):
                order=sorted(range(len(module.experts)),key=lambda e:(placements[index][e],e))
                original_experts=module.experts
                original_forward=module.forward
                inverse=[order.index(logical) for logical in range(len(order))]
                saved.append((module,original_experts,original_forward))
                module.experts=nn.ModuleDict({f'expert_{i}':original_experts[f'expert_{e}'] for i,e in enumerate(order)})
                def forward(self,hidden_states,mapping=inverse):
                    router_mask,router_probs,router_logits=self.router(hidden_states)
                    expert_index=router_mask.argmax(-1)
                    next_states=hidden_states.clone()
                    for logical,physical in enumerate(mapping):
                        selected=router_mask[:,:,logical].bool()
                        next_states[selected]=self.experts[f'expert_{physical}'](hidden_states[selected]).to(next_states.dtype)
                    return router_probs*next_states,(router_logits,expert_index)
                module.forward=MethodType(forward,module)
        yield
    finally:
        with torch.no_grad():
            for module,experts,forward in saved:
                module.experts=experts;module.forward=forward


def greedy_placement(counts,mode,devices=4,capacity=2):
    """Counts[window,origin,expert,layer]; admissibility checked at each insertion."""
    total=counts.sum(0);result=[];details=[]
    for layer in range(total.shape[-1]):
        traffic=total[:,:,layer];loads=traffic.sum(0);maximum=1.25*float(loads.sum())/devices
        device_load=[0.]*devices;slots=[0]*devices;placement=[-1]*len(loads)
        for expert in sorted(range(len(loads)),key=lambda e:(-float(loads[e]),e)):
            valid=[d for d in range(devices) if slots[d]<capacity and device_load[d]+float(loads[expert])<=maximum+1e-6]
            if not valid:return None,{'admissible':False,'layer':layer,'expert':expert,'reason':'no load/capacity admissible greedy destination'}
            def cost(destination):
                return sum(float(traffic[origin,expert])*min((origin-destination)%devices,(destination-origin)%devices) for origin in range(devices))
            chosen=min(valid,key=lambda d:((device_load[d] if mode=='balanced' else cost(d)),d))
            placement[expert]=chosen;device_load[chosen]+=float(loads[expert]);slots[chosen]+=1
        result.append(placement);details.append({'device_load':device_load,'maximum_load':maximum,'slots':slots})
    return result,{'admissible':True,'layers':details}


def validate_actual_load(counts,placements,devices=4):
    details=[]
    for layer,placement in enumerate(placements):
        loads=counts[:,:,:,layer].sum((0,1))
        device=[sum(float(loads[e]) for e,d in enumerate(placement) if d==destination) for destination in range(devices)]
        maximum=1.25*sum(device)/devices
        details.append({'load':device,'maximum':maximum,'admissible':max(device)<=maximum+1e-6})
    return all(record['admissible'] for record in details),details
