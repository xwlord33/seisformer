"""Localize mixed-precision instability on a saved training state and train batch."""
from pathlib import Path
import gc
import json
import sys
import traceback

import torch
from torch.utils.data import DataLoader

from scripts.recent_dual_core import PROJECT,read,sha,build_model,MatchedWaveforms,PairedEpochSampler,forward,matched_loss
from scripts.diagnose_recent_dual_nan import float_modules
from scripts.rebuild_standardized import atomic_json
from seisformer.engine import seed_all

ROOT=PROJECT/'outputs/recent_dual_comparison_v1'


def run():
    from scripts import recent_dual_core as core
    # This diagnostic intentionally reconstructs the original BF16-only control;
    # the production adapter already installs the verified FP32 LSTM guard.
    core.fp32_recurrent=lambda model:[]
    output=ROOT/'precision_replay';output.mkdir(exist_ok=True)
    checkpoint=torch.load(ROOT/'diagnostic_bf16_attempt/training/twostream_eqt_s42/last.pt',map_location='cpu',weights_only=False)
    seed_all(42);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sampler=PairedEpochSampler(262144,42);sampler.epoch=checkpoint['epoch']
    failure_path=ROOT/'nan_diagnosis/failure.pt'
    if failure_path.exists():
        failure=torch.load(failure_path,map_location='cpu',weights_only=False)
        batch=failure['batch'];checkpoint['model']=failure['model'];checkpoint['cuda_rng']=failure['cuda_rng']
    else:batch=next(iter(DataLoader(MatchedWaveforms(ROOT/'cache/train'),sampler=sampler,batch_size=512,num_workers=0)))
    torch.save(batch,output/'probe_batch.pt')
    records=[]
    for label,kinds in [('bf16',set()),('normalization_fp32',{'LayerNormalization'}),('lstm_fp32',{'LSTM'}),('both_fp32',{'LayerNormalization','LSTM'}),('all_fp32',set())]:
        model=out=loss=None
        try:
            model=build_model('twostream_eqt').cuda().train();model.load_state_dict(checkpoint['model'],strict=True)
            changed=float_modules(model,kinds);torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=label!='all_fp32'):
                out=forward(model,'twostream_eqt',batch);loss=matched_loss(out,batch)
            loss.backward()
            bad=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            rec=dict(variant=label,loss=float(loss.detach()) if torch.isfinite(loss) else None,finite_loss=bool(torch.isfinite(loss)),
                nonfinite_gradient_parameters=bad,finite_gradients=not bad,changed_modules=changed,
                output_finite={key:bool(torch.isfinite(value).all()) for key,value in out.items()})
            if label=='bf16' and bad:
                model.zero_grad(set_to_none=True);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
                try:
                    with torch.autograd.detect_anomaly():
                        with torch.autocast('cuda',dtype=torch.bfloat16):value=matched_loss(forward(model,'twostream_eqt',batch),batch)
                        value.backward()
                except Exception as anomaly:rec['backward_anomaly']=str(anomaly)
        except Exception as e:rec=dict(variant=label,error=str(e),traceback=traceback.format_exc())
        records.append(rec);atomic_json(output/'precision_probe.json',dict(records=records,input_finite=bool(torch.isfinite(batch['x']).all()),checkpoint_epoch=checkpoint['epoch']))
        print(json.dumps(rec),flush=True)
        del model,out,loss
        module=sys.modules.get('seisbench.models.recent_dual_crosstwostreameqt')
        if module is not None:module.xf=None
        gc.collect();torch.cuda.empty_cache()
    atomic_json(output/'precision_probe.json',dict(status='complete',records=records,input_finite=bool(torch.isfinite(batch['x']).all()),
        checkpoint_epoch=checkpoint['epoch'],input_source='Training only; first paired epoch7 batch.',upstream_network_modified=False))


if __name__=='__main__':run()
