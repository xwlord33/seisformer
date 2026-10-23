"""Train one immutable job in the matched recent dual-domain experiment."""
import argparse
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.recent_dual_core import (PROJECT,read,sha,build_model,MatchedWaveforms,PairedEpochSampler,
    forward,matched_loss,evaluate,operating_point,calibrated_metrics)
from scripts.rebuild_standardized import atomic_json
from seisformer.engine import seed_all,worker_seed


def verify(config):
    for key in ('input_sha256','source_sha256'):
        for path,value in config.get(key,{}).items():
            if sha(path)!=value:raise ValueError('Frozen dependency changed: '+path)


def atomic_checkpoint(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)


def run(args):
    config=read(args.config);verify(config)
    root=Path(args.config).resolve().parent;out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise ValueError('Choose a fresh job directory; existing runs are preserved.')
    seed_all(args.seed);torch.set_num_threads(config['threads']);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    training=MatchedWaveforms(root/'cache/train',config['tune_records'] if args.tuning else 0,data_seed=config['data_seed'])
    validation=MatchedWaveforms(root/'cache/val',config['tune_validation_records'] if args.tuning else 0,data_seed=42)
    sampler=PairedEpochSampler(len(training),args.seed)
    loader_args=dict(batch_size=config['batch_size'],num_workers=config['workers'],pin_memory=True,
        persistent_workers=config['workers']>0,worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(args.seed))
    train_loader=DataLoader(training,sampler=sampler,**loader_args)
    val_loader=DataLoader(validation,shuffle=False,**loader_args)
    model=build_model(args.architecture).cuda()
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters=={'seisformer':428774,'twostream_eqt':622668}[args.architecture]
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=config['weight_decay'])
    epochs=config['tune_epochs'] if args.tuning else config['epochs']
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*len(train_loader),eta_min=args.lr*config['minimum_lr_factor'])
    initial_sha={key:sha(path) for key,path in [('config',args.config)]}
    runinfo=dict(architecture=args.architecture,seed=args.seed,lr=args.lr,tuning=args.tuning,
        config_sha256=initial_sha['config'],parameters=parameters,epochs=epochs,train_records=len(training),
        validation_records=len(validation),batch_size=config['batch_size'],updates_per_epoch=len(train_loader),
        initialization='fresh_random_from_scratch',upstream_source_sha256=sha(PROJECT/'third_party/twostream_eqt_2025/cross_twostreameqt.py'))
    atomic_json(out/'run.json',runinfo);best=float('inf');global_step=0;started=time.perf_counter()
    for epoch in range(epochs):
        sampler.epoch=epoch;model.train();total_loss=0.;seen=0;epoch_started=time.perf_counter()
        for step,batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=matched_loss(forward(model,args.architecture,batch),batch)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();global_step+=1
            total_loss+=float(loss.detach())*len(batch['x']);seen+=len(batch['x'])
            if (step+1)%50==0:
                atomic_json(out/'progress.json',dict(status='training',architecture=args.architecture,seed=args.seed,
                    epoch=epoch+1,epochs=epochs,step=step+1,steps_per_epoch=len(train_loader),
                    samples_per_second=seen/(time.perf_counter()-epoch_started),loss=total_loss/seen,
                    elapsed_seconds=time.perf_counter()-started,global_step=global_step))
        train_seconds=time.perf_counter()-epoch_started
        atomic_json(out/'progress.json',dict(status='validating',epoch=epoch+1,epochs=epochs,global_step=global_step))
        metrics,predictions=evaluate(model,args.architecture,val_loader,bf16=True)
        score=float(np.mean([metrics['overall'][p]['mae_s'] for p in ('P','S')]))
        improved=score<best;best=min(best,score)
        record=dict(epoch=epoch+1,train_loss=total_loss/seen,train_seconds=train_seconds,
            mean_ps_mae_s=score,P_mae_s=metrics['overall']['P']['mae_s'],S_mae_s=metrics['overall']['S']['mae_s'],
            detection_f1=metrics['overall']['detection']['0.5']['f1'],global_step=global_step,
            elapsed_seconds=time.perf_counter()-started,selected=improved)
        with (out/'history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        checkpoint=dict(format_version=config['version'],architecture=args.architecture,model=model.state_dict(),
            seed=args.seed,lr=args.lr,epoch=epoch+1,global_step=global_step,best=best,run=runinfo,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state())
        atomic_checkpoint(out/'last.pt',checkpoint)
        if improved:
            atomic_checkpoint(out/'best.pt',checkpoint);atomic_json(out/'best_validation.json',record)
        verify(config)
        print(json.dumps(dict(job=out.name,**record)),flush=True)
    del checkpoint
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'],strict=True)
    metrics,predictions=evaluate(model,args.architecture,val_loader,bf16=False)
    np.savez_compressed(out/'validation_predictions.npz',**predictions)
    point=operating_point(predictions,.01)
    atomic_json(out/'validation_metrics.json',dict(**metrics,operating_point=point,
        at_validation_operating_point=calibrated_metrics(predictions,point)))
    result=dict(status='complete',architecture=args.architecture,seed=args.seed,lr=args.lr,
        completed_epochs=epochs,completed_updates=global_step,selected_epoch=selected['epoch'],
        selected_checkpoint_sha256=sha(out/'best.pt'),validation_mae=best,
        validation_fp32_mean_ps_mae_s=float(np.mean([metrics['overall'][p]['mae_s'] for p in ('P','S')])),
        elapsed_seconds=time.perf_counter()-started,parameters=parameters,
        config_sha256=sha(args.config),training_from_scratch=True)
    atomic_json(out/'completion.json',result);atomic_json(out/'progress.json',result)
    print(json.dumps(dict(job=out.name,**result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--architecture',required=True,choices=['seisformer','twostream_eqt'])
    p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--tuning',action='store_true')
    args=p.parse_args()
    try:run(args)
    except Exception:
        target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
        atomic_json(target/'progress.json',dict(status='failed',traceback=traceback.format_exc()));raise
