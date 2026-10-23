"""Run matched LR screens, all six trainings, then the frozen evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

from scripts.recent_dual_core import PROJECT,read,sha
from scripts.rebuild_standardized import atomic_json
from scripts.train_recent_dual_comparison import verify
from scripts.analyze_recent_dual_comparison import verify_completion,run as analyze


def run(config_path):
    config_path=Path(config_path).resolve();cfg=read(config_path);verify(cfg);root=config_path.parent
    owner_path=root/'campaign.owner.json'
    if owner_path.exists():
        owner=read(owner_path)
        try:
            process=psutil.Process(owner['pid'])
            if abs(process.create_time()-owner['creation_time'])<.01:
                raise RuntimeError('This exact campaign process is already active.')
        except psutil.NoSuchProcess:pass
    atomic_json(owner_path,dict(pid=os.getpid(),creation_time=psutil.Process().create_time(),config_sha256=sha(config_path)))
    started=time.time();completed=[]
    def job(architecture,seed,lr,tuning=False):
        directory=root/('tuning' if tuning else 'training')/(f'{architecture}_lr{lr:g}' if tuning else f'{architecture}_s{seed}')
        epochs=cfg['tune_epochs'] if tuning else cfg['epochs']
        if (directory/'completion.json').exists():
            result=verify_completion(directory,cfg,architecture,seed,epochs)
            if result['config_sha256']!=sha(config_path):
                compatible=cfg.get('compatible_completed_jobs',{}).get(directory.name,{})
                assert compatible.get('architecture')==architecture=='seisformer'
                assert compatible.get('config_sha256')==result['config_sha256']
            return result
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'run.json').exists():
            raise RuntimeError('An incomplete job is preserved; inspect it before recovery: '+str(directory))
        command=[sys.executable,'-X','utf8','-u','-m','scripts.train_recent_dual_comparison',
            '--config',str(config_path),'--output',str(directory),'--architecture',architecture,
            '--seed',str(seed),'--lr',str(lr)]
        if tuning:command.append('--tuning')
        atomic_json(root/'campaign.progress.json',dict(status='tuning' if tuning else 'training',
            architecture=architecture,seed=seed,lr=lr,current_job=str(directory),
            completed_jobs=completed,elapsed_seconds=time.time()-started))
        with (directory/'stdout.log').open('a',encoding='utf-8') as stdout,(directory/'stderr.log').open('a',encoding='utf-8') as stderr:
            process=subprocess.Popen(command,cwd=PROJECT,stdout=stdout,stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            atomic_json(directory/'process.json',dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),command=command))
            returncode=process.wait()
        if returncode:raise RuntimeError(f'Job failed (exit {returncode}): {directory}; inspect stderr.log and progress.json.')
        verify(cfg);result=verify_completion(directory,cfg,architecture,seed,epochs)
        completed.append(dict(directory=str(directory),architecture=architecture,seed=seed,tuning=tuning,
            validation_fp32_mean_ps_mae_s=result['validation_fp32_mean_ps_mae_s']))
        print(json.dumps(dict(completed_job=str(directory),**result)),flush=True)
        return result
    selected={}
    for architecture in cfg['architectures']:
        screens=[]
        for lr in cfg['learning_rates']:
            result=job(architecture,42,lr,tuning=True)
            screens.append(dict(learning_rate=lr,validation_fp32_mean_ps_mae_s=result['validation_fp32_mean_ps_mae_s'],
                selected_checkpoint_sha256=result['selected_checkpoint_sha256']))
        winner=min(screens,key=lambda r:(r['validation_fp32_mean_ps_mae_s'],r['learning_rate']))
        selected[architecture]=dict(learning_rate=winner['learning_rate'],candidates=screens,
            criterion='FP32 mean P/S MAE of validation-selected screen checkpoint; fixed shared validation subset')
        atomic_json(root/'selected_learning_rates.json',selected)
    for index,seed in enumerate(cfg['seeds']):
        order=cfg['architectures'] if index%2==0 else list(reversed(cfg['architectures']))
        for architecture in order:job(architecture,seed,selected[architecture]['learning_rate'])
    atomic_json(root/'campaign.progress.json',dict(status='evaluating',all_six_trainings_complete=True,
        completed_jobs=completed,elapsed_seconds=time.time()-started))
    result=analyze(config_path)
    atomic_json(root/'campaign.progress.json',dict(status='complete',all_six_trainings_complete=True,
        elapsed_seconds=time.time()-started,result_sha256=sha(root/'RESULTS.json'),summary=result['summary']))
    print(json.dumps(dict(status='complete',elapsed_seconds=time.time()-started,summary=result['summary'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    try:run(args.config)
    except Exception:
        root=Path(args.config).resolve().parent
        atomic_json(root/'campaign.progress.json',dict(status='failed',traceback=traceback.format_exc()));raise
