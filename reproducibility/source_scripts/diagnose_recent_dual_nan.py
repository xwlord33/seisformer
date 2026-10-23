"""Replay a failed training epoch from its last complete checkpoint, without test access."""
from pathlib import Path
import types
import json
import time

import torch
import numpy as np
from torch.utils.data import DataLoader

from scripts.recent_dual_core import PROJECT,read,build_model,MatchedWaveforms,PairedEpochSampler,forward,matched_loss
from scripts.rebuild_standardized import atomic_json
from seisformer.engine import seed_all,worker_seed

ROOT=PROJECT/'outputs/recent_dual_comparison_v1'


def float_modules(model,kinds):
    changed=[]
    for name,module in model.named_modules():
        if module.__class__.__name__ in kinds:
            original=module.forward
            def stable(*args,_original=original,**kwargs):
                def cast(v):
                    if torch.is_tensor(v) and v.is_floating_point():return v.float()
                    if isinstance(v,tuple):return tuple(cast(x) for x in v)
                    return v
                with torch.autocast('cuda',enabled=False):return _original(*(cast(a) for a in args),**kwargs)
            module.forward=stable;changed.append(name)
    return changed


def run():
    from scripts import recent_dual_core as core
    core.fp32_recurrent=lambda model:[]
    cfg=read(ROOT/'config.json');directory=ROOT/'diagnostic_bf16_attempt/training/twostream_eqt_s42';out=ROOT/'precision_replay_diagnosis';out.mkdir(exist_ok=True)
    checkpoint=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    seed_all(42);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=build_model('twostream_eqt').cuda().train();model.load_state_dict(checkpoint['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);optimizer.load_state_dict(checkpoint['optimizer'])
    sampler=PairedEpochSampler(262144,42);sampler.epoch=checkpoint['epoch']
    dataset=MatchedWaveforms(ROOT/'cache/train')
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],sampler=sampler,num_workers=cfg['workers'],
        persistent_workers=True,pin_memory=True,worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(42))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20*512,eta_min=.0001);scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    started=time.time()
    for step,batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True);rng=torch.cuda.get_rng_state_all()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=forward(model,'twostream_eqt',batch);loss=matched_loss(result,batch)
        if torch.isfinite(loss):loss.backward()
        nonfinite_gradients=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if not torch.isfinite(loss) or nonfinite_gradients:
            torch.save(dict(batch=batch,model=model.state_dict(),cuda_rng=rng,step=step,epoch=sampler.epoch+1),out/'failure.pt')
            first=[];handles=[]
            def check(name):
                def hook(module,inputs,outputs):
                    values=outputs if isinstance(outputs,tuple) else (outputs,)
                    tensors=[v for v in values if torch.is_tensor(v)]
                    if tensors and any(not torch.isfinite(v).all() for v in tensors):first.append(dict(name=name,kind=module.__class__.__name__))
                return hook
            for name,module in model.named_modules():handles.append(module.register_forward_hook(check(name)))
            torch.cuda.set_rng_state_all(rng)
            with torch.autocast('cuda',dtype=torch.bfloat16):forward(model,'twostream_eqt',batch)
            for h in handles:h.remove()
            variants=[]
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            del model,optimizer,result,loss
            import sys,gc
            sys.modules['seisbench.models.recent_dual_crosstwostreameqt'].xf=None
            gc.collect();torch.cuda.empty_cache()
            for label,kinds in [('normalization_fp32',{'LayerNormalization'}),('lstm_fp32',{'LSTM'}),('both_fp32',{'LayerNormalization','LSTM'}),('all_fp32',set())]:
                candidate=build_model('twostream_eqt').cuda().train();candidate.load_state_dict(state,strict=True)
                changed=float_modules(candidate,kinds);torch.cuda.set_rng_state_all(rng)
                try:
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=label!='all_fp32'):
                        value=matched_loss(forward(candidate,'twostream_eqt',batch),batch)
                    value.backward()
                    finite_grads=all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
                    variants.append(dict(label=label,finite_loss=bool(torch.isfinite(value)),finite_gradients=bool(finite_grads),loss=float(value.detach()) if torch.isfinite(value) else None,changed=changed))
                except Exception as e:variants.append(dict(label=label,error=str(e)))
                del candidate;torch.cuda.empty_cache()
            report=dict(status='reproduced',epoch=sampler.epoch+1,step=step+1,input_finite=bool(torch.isfinite(batch['x']).all()),
                max_input_amplitude=float(batch['x'].abs().max()),first_nonfinite_modules=first[:20],
                nonfinite_gradient_parameters=nonfinite_gradients,variants=variants,elapsed_seconds=time.time()-started)
            atomic_json(out/'diagnosis.json',report);print(json.dumps(report,indent=2));return
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        if (step+1)%50==0:print('replay_step',step+1,'loss',float(loss.detach()),flush=True)
    atomic_json(out/'diagnosis.json',dict(status='not_reproduced_with_one_epoch_replay',epoch=sampler.epoch+1,elapsed_seconds=time.time()-started))


if __name__=='__main__':run()
