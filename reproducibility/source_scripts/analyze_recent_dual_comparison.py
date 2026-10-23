"""Evaluate frozen matched checkpoints and produce the requested comparison table."""
from pathlib import Path
import gc
import json
import math
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.recent_dual_core import (PROJECT,read,sha,make_cache,MatchedWaveforms,build_model,
    evaluate,calibrated_metrics)
from scripts.rebuild_standardized import atomic_json
from scripts.train_recent_dual_comparison import verify


def verify_completion(directory,cfg,architecture,seed,epochs):
    directory=Path(directory);record=read(directory/'completion.json');run=read(directory/'run.json')
    assert record['status']=='complete' and record['completed_epochs']==epochs
    assert record['architecture']==architecture and record['seed']==seed
    assert record['completed_updates']==epochs*math.ceil(run['train_records']/cfg['batch_size'])
    assert record['selected_checkpoint_sha256']==sha(directory/'best.pt')
    history=[json.loads(line) for line in (directory/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(history)==epochs and [v['epoch'] for v in history]==list(range(1,epochs+1))
    assert 1<=record['selected_epoch']<=epochs and run['initialization']=='fresh_random_from_scratch'
    return record


def paired_bootstrap(root,records,replicates=2000):
    baseline=np.load(Path(records[0]['directory'])/'test_predictions.npz',allow_pickle=False)
    groups=baseline['group'];unique,group_id=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(20261022);result={}
    for j,phase in enumerate(('P','S')):
        known=(baseline['truth'][:,j]>=0)&baseline['event_mask']
        active=np.unique(group_id[known]);mapping=np.full(len(unique),-1,int);mapping[active]=np.arange(len(active))
        ids=mapping[group_id[known]];count=np.bincount(ids,minlength=len(active))
        sums={a:[] for a in ('seisformer','twostream_eqt')}
        for architecture in sums:
            for seed in (42,43,44):
                record=next(v for v in records if v['architecture']==architecture and v['seed']==seed)
                with np.load(Path(record['directory'])/'test_predictions.npz',allow_pickle=False) as a:
                    error=np.abs(a['pred'][known,j]-a['truth'][known,j])/100
                sums[architecture].append(np.bincount(ids,weights=error,minlength=len(active)))
        sums={a:np.asarray(v) for a,v in sums.items()};differences=[];ratios=[]
        for rep in range(replicates):
            seeds=rng.integers(0,3,3);weights=np.bincount(rng.integers(0,len(active),len(active)),minlength=len(active))
            estimates={a:float(v[seeds].mean(0)@weights/(count@weights)) for a,v in sums.items()}
            differences.append(estimates['seisformer']-estimates['twostream_eqt'])
            ratios.append(estimates['seisformer']/max(estimates['twostream_eqt'],1e-12))
        result[phase]=dict(groups=len(active),records=int(known.sum()),replicates=replicates,
            difference_seisformer_minus_twostream_s_ci95=np.quantile(differences,[.025,.975]).tolist(),
            mae_ratio_seisformer_over_twostream_ci95=np.quantile(ratios,[.025,.975]).tolist(),
            resampling='paired model seeds and event groups; within-group record weights preserved')
    return result


@torch.inference_mode()
def benchmark(models):
    """Alternating blocks on the same device, input and precision."""
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for device,batches in (('cuda',(1,64)),('cpu',(1,))):
        for batch in batches:
            torch.manual_seed(42);x=torch.randn(batch,3,6000,device=device)
            for name,model in models.items():
                model.to(device).eval()
                for warmup in range(10):model(x)
            if device=='cuda':torch.cuda.synchronize()
            timings={name:[] for name in models}
            for repeat in range(3):
                names=list(models) if repeat%2==0 else list(reversed(models))
                for name in names:
                    model=models[name]
                    if device=='cuda':torch.cuda.synchronize()
                    started=time.perf_counter()
                    for iteration in range(20):out=model(x)
                    if device=='cuda':torch.cuda.synchronize()
                    assert all(v.device.type==device for v in out.values())
                    timings[name].append((time.perf_counter()-started)/20)
            for name,values in timings.items():
                seconds=float(np.mean(values))
                results.append(dict(architecture=name,device=device,batch=batch,precision='FP32',
                    milliseconds_per_batch=1000*seconds,windows_per_second=batch/seconds,
                    block_seconds_per_batch=values,warmup=10,measured_iterations=60))
            del x
        for model in models.values():model.cpu()
        gc.collect();torch.cuda.empty_cache()
    return dict(gpu=torch.cuda.get_device_name(),cpu=platform.processor(),cpu_threads=2,
        torch_version=torch.__version__,TF32=False,results=results,
        scope='Same forward call, native spectral paths and TwoStream preprocessing included; input cache/disk/transfers excluded. No streaming-latency claim.')


def run(config_path):
    cfg=read(config_path);verify(cfg);root=Path(config_path).resolve().parent
    torch.set_num_threads(cfg['threads']);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selected=read(root/'selected_learning_rates.json');runs=[]
    for seed in cfg['seeds']:
        for architecture in cfg['architectures']:
            directory=root/'training'/f'{architecture}_s{seed}'
            completed=verify_completion(directory,cfg,architecture,seed,cfg['epochs'])
            assert completed['lr']==selected[architecture]['learning_rate']
            runs.append(dict(directory=str(directory),architecture=architecture,seed=seed,completion=completed))
    atomic_json(root/'evaluation_progress.json',dict(status='building_test_cache',all_six_training_jobs_verified=True))
    entries=read(cfg['test_rows'])
    make_cache(entries,cfg['data'],cfg['test_snapshot'],'test',root/'cache/test')
    test=MatchedWaveforms(root/'cache/test',data_seed=42)
    loader=DataLoader(test,batch_size=64,num_workers=cfg['workers'],persistent_workers=cfg['workers']>0,pin_memory=True)
    reference_identity=None
    for record in runs:
        directory=Path(record['directory']);architecture=record['architecture']
        model=build_model(architecture).cuda().eval()
        checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True)
        metrics,predictions=evaluate(model,architecture,loader,bf16=False)
        identity={k:predictions[k] for k in ('truth','event','source','event_mask','row','group','crop_start')}
        if reference_identity is None:reference_identity=identity
        else:
            for key,value in identity.items():assert np.array_equal(value,reference_identity[key]),key
        assert metrics['overall']['n']==116420 and metrics['overall']['P']['annotated_n']==94278 and metrics['overall']['S']['annotated_n']==79255
        point=read(directory/'validation_metrics.json')['operating_point']
        metrics.update(checkpoint_sha256=record['completion']['selected_checkpoint_sha256'],selected_epoch=checkpoint['epoch'],
            architecture=architecture,seed=record['seed'],operating_point=point,
            at_validation_operating_point=calibrated_metrics(predictions,point))
        np.savez_compressed(directory/'test_predictions.npz',**predictions)
        atomic_json(directory/'test_metrics.json',metrics);record['metrics']=metrics
        atomic_json(root/'evaluation_progress.json',dict(status='evaluating',completed=len([v for v in runs if 'metrics' in v]),total=len(runs),last=dict(architecture=architecture,seed=record['seed'])))
        del checkpoint,model;gc.collect();torch.cuda.empty_cache()
    inference_models={}
    for architecture in cfg['architectures']:
        directory=root/'training'/f'{architecture}_s42'
        model=build_model(architecture).eval()
        model.load_state_dict(torch.load(directory/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True)
        inference_models[architecture]=model
    timing=benchmark(inference_models);atomic_json(root/'runtime.json',timing)
    intervals=paired_bootstrap(root,runs)
    summary={}
    for architecture in cfg['architectures']:
        group=[r for r in runs if r['architecture']==architecture]
        summary[architecture]=dict(parameters=group[0]['completion']['parameters'],seeds=cfg['seeds'],
            learning_rate=selected[architecture]['learning_rate'],
            **{phase:dict(mae_s_mean=float(np.mean([r['metrics']['overall'][phase]['mae_s'] for r in group])),
                mae_s_sd=float(np.std([r['metrics']['overall'][phase]['mae_s'] for r in group],ddof=1)),
                calibrated_f1_mean=float(np.mean([r['metrics']['at_validation_operating_point'][phase]['f1'] for r in group])),
                calibrated_f1_sd=float(np.std([r['metrics']['at_validation_operating_point'][phase]['f1'] for r in group],ddof=1))) for phase in ('P','S')})
    result=dict(status='complete',config_sha256=sha(config_path),summary=summary,paired_intervals=intervals,
        runs=runs,runtime=timing,all_six_test_metadata_arrays_match=True,all_six_training_jobs_verified=True,
        checkpoint_selection_used_test=False,threshold_selection_used_test=False,
        description='Matched 262k/20-epoch training from scratch; not a full reproduction of upstream STEAD 100-epoch training.')
    atomic_json(root/'RESULTS.json',result)
    def runtime(architecture,device,batch,key):return next(r[key] for r in timing['results'] if r['architecture']==architecture and r['device']==device and r['batch']==batch)
    rows=[]
    for architecture,label in (('seisformer','SeisFormer'),('twostream_eqt','TwoStream-EQT（C&G 2025）')):
        r=summary[architecture]
        rows.append(f"| {label} | {r['P']['mae_s_mean']:.4f} ± {r['P']['mae_s_sd']:.4f} | {r['S']['mae_s_mean']:.4f} ± {r['S']['mae_s_sd']:.4f} | {r['P']['calibrated_f1_mean']:.4f} | {r['S']['calibrated_f1_mean']:.4f} | {r['parameters']:,} | {runtime(architecture,'cuda',1,'milliseconds_per_batch'):.2f} | {runtime(architecture,'cuda',64,'windows_per_second'):,.0f} | {runtime(architecture,'cpu',1,'milliseconds_per_batch'):.2f} |")
    text='''# 近期双域模型公平对照结果

两种模型均从随机初始化开始，使用相同262,144条训练记录、16,384条验证记录和127,940条测试清单；3个种子、各20轮，相同批次和裁窗序列。均给予相同的3候选学习率筛选预算。以下为三种子均值，MAE同时给出种子间样本标准差。

| 模型 | P MAE（秒） | S MAE（秒） | P F1 | S F1 | 参数量 | GPU B1（ms/窗） | GPU B64（窗/秒） | CPU B1（ms/窗） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

F1容差为±0.3秒；阈值仅在验证集上选定，两个模型共同采用1%验证噪声预算。MAE对全部已知到时计算，不筛选正确检测或高评分样本。固定0.3震相/0.5事件阈值的原始结果也保留在各模型test_metrics.json中。

参数量计入完整公开网络，包括保留的原生检测头。FP32效率使用同一RTX 5070 Ti、同一输入与交替计时顺序；CPU为两线程。包含频谱计算和TwoStream-EQT原生预处理，排除磁盘与设备传输。公共训练batch为{cfg['batch_size']}；学习率由各自验证筛选决定：SeisFormer {selected['seisformer']['learning_rate']:g}，TwoStream-EQT {selected['twostream_eqt']['learning_rate']:g}。

## 配对到时差值的不确定性

差值方向为SeisFormer减TwoStream-EQT；同时重采样模型种子与事件组，保留组内记录权重。
'''
    for phase in ('P','S'):
        lo,hi=intervals[phase]['difference_seisformer_minus_twostream_s_ci95'];ratio=intervals[phase]['mae_ratio_seisformer_over_twostream_ci95']
        text+=f"\n- {phase}：MAE差值95%区间 [{lo:.4f}, {hi:.4f}]秒；MAE比值区间 [{ratio[0]:.3f}, {ratio[1]:.3f}]。\n"
    text+='''
## 与原1M最佳权重的关系

历史1M最佳模型的P/S MAE为0.266/0.216秒、检测F1为0.9988。它包含先导和524k继续训练历史，使用不同的训练规模，作为独立背景成绩保留。上表两行只比较新训练的同预算模型，不把历史最佳权重与从零训练的262k基线混排成公平胜负。

## 实现与比较范围

TwoStream-EQT采用作者公开的交叉双流实现（7层卷积、7层残差、3层LSTM、STFT与交叉注意力），源码与上游blob逐字节核对；保留ZNE/std/六点taper预处理。P/S目标、损失、可用性掩码和取峰规则共享。检测辅助任务保留各自输出结构：SeisFormer为窗口BCE，TwoStream-EQT为P至P+1.4(S-P)的时序BCE，未知完整事件区间不作为静噪监督。

这项实验比较相同数据预算下的两种实现，不复刻原论文独立STEAD/100轮/增强配方，也不据此建立跨论文排行榜。所有种子和结果均保留。详情见PROTOCOL.md、config.json、selected_learning_rates.json、各运行history.jsonl及完整RESULTS.json。
'''
    (root/'RESULTS.md').write_text(text,encoding='utf-8')
    paper=PROJECT/'outputs/paper_ps'
    (paper/'近期双域公平对照结果.md').write_text(text,encoding='utf-8')
    atomic_json(root/'evaluation_progress.json',dict(status='complete',result_sha256=sha(root/'RESULTS.json')))
    print(json.dumps(dict(status='complete',summary=summary,paired_intervals=intervals),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
