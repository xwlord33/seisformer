"""Measure common training batch sizes and freeze the matched protocol."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.recent_dual_core import PROJECT,MatchedWaveforms,build_model,forward,matched_loss,read,sha
from scripts.rebuild_standardized import atomic_json
from seisformer.engine import seed_all

ROOT=PROJECT/'outputs/recent_dual_comparison_v1'


def run():
    if (ROOT/'config.json').exists():raise ValueError('Protocol already frozen; preserve the original preflight.')
    cfg=read(ROOT/'config.preflight.json');seed_all(42);torch.set_num_threads(cfg['threads'])
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=MatchedWaveforms(ROOT/'cache/train',limit=512)
    cpu=next(iter(DataLoader(data,batch_size=512,num_workers=0)))
    results=[]
    for architecture in cfg['architectures']:
        for batch_size in (64,256,512):
            seed_all(42);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model=optimizer=out=loss=batch=None
            try:
                model=build_model(architecture).cuda().train()
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
                batch={k:v[:batch_size].cuda() if torch.is_tensor(v) else v[:batch_size] for k,v in cpu.items()}
                elapsed=[]
                for step in range(16):
                    torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=matched_loss(forward(model,architecture,batch),batch)
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    torch.cuda.synchronize();elapsed.append(time.perf_counter()-started)
                results.append(dict(architecture=architecture,batch_size=batch_size,status='passed',
                    samples_per_second=batch_size/float(np.mean(elapsed[8:])),step_seconds=float(np.mean(elapsed[8:])),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,parameters=sum(p.numel() for p in model.parameters()),
                    finite_loss_and_gradients=True))
            except torch.cuda.OutOfMemoryError:
                results.append(dict(architecture=architecture,batch_size=batch_size,status='oom'))
            finally:
                del model,optimizer,out,loss,batch;gc.collect();torch.cuda.empty_cache()
            atomic_json(ROOT/'gpu_preflight.json',dict(status='running',results=results))
    common=[]
    for batch in (64,256,512):
        group=[r for r in results if r['batch_size']==batch and r['status']=='passed']
        if len(group)==2:common.append((float(np.prod([v['samples_per_second'] for v in group])**.5),batch))
    assert common,'No common feasible training batch size'
    chosen=max(common)[1];cfg['batch_size']=chosen
    paths=[PROJECT/'scripts'/f'{name}.py' for name in ('recent_dual_core','prepare_recent_dual_comparison',
        'preflight_recent_dual_comparison','train_recent_dual_comparison','run_recent_dual_comparison','analyze_recent_dual_comparison')]
    paths += [PROJECT/'seisformer'/f'{name}.py' for name in ('model','engine','metrics','standardized')]
    paths += [PROJECT/'third_party/twostream_eqt_2025'/name for name in ('cross_twostreameqt.py','PROVENANCE.json')]
    cfg['source_sha256']={str(p):sha(p) for p in paths}
    cfg['batch_selection']='Largest measured geometric-mean training throughput among mutually feasible B64/B256/B512; no accuracy/test criteria.'
    atomic_json(ROOT/'gpu_preflight.json',dict(status='passed',results=results,common_batch_size=chosen,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__))
    cfg['gpu_preflight_sha256']=sha(ROOT/'gpu_preflight.json')
    atomic_json(ROOT/'config.json',cfg)
    protocol=f'''# 近期双域模型配对比较：预先固定方案

比较SeisFormer与C&G 2025 TwoStream-EQT公开交叉双流实现。两者均从随机权重训练，使用相同262,144条训练记录、16,384条验证记录和127,940条测试清单。相同100 Hz、60秒物理窗口；每个种子、轮次和记录的裁窗与批次顺序跨结构配对。

先在相同32k训练子集和4k验证子集上分别检验三个学习率（1e-4、3e-4、1e-3），每个3轮。按验证平均P/S MAE选各自学习率。正式训练两种结构×三个种子（42/43/44），各20轮，公共batch {chosen}，AdamW、余弦调度、梯度裁剪和P/S标签/损失一致。无预训练历史，也不从调参模型继续。每轮按共同完整验证集的平均P/S MAE选权重。

TwoStream-EQT结构保留原始7层卷积/7层残差/3层LSTM和交叉注意力，源码逐字节校验，原生ZNE/std/六点taper预处理保留。其检测头沿时间输出，按P到P+1.4(S-P)区间监督；只有完整已知P/S对或确认噪声才监督检测曲线。SeisFormer保留原生窗口检测头。两种检测头的标签形状不同，报告中明确。这是相同数据预算下的任务适配，不冒充原论文STEAD 100轮训练的复刻。

训练、学习率和选模仅访问train/val。所有正式权重选定后才评价固定测试清单。主要精度为所有已知目标的无条件P/S MAE；同时报告±0.3秒拾取F1、固定阈值结果，以及在验证噪声1%预算下选阈值的结果。未知事件震相不计作负例，错误拾取同时计FP和FN。

效率用同一GPU、CPU两线程、FP32和相同输入/批次，包含原生频谱及网络预处理，排除磁盘和传输。报告B1延迟、B64吞吐和完整参数量。历史1M最佳权重保留为不同训练阶段的背景行，不用于匹配预算下的胜负结论。

本方案在正式训练及测试前固定。实际冻结配置、源码和输入哈希见config.json。数据缓存通过与原始读器的实际样本逐字段一致性检查。任何结果均保留，不按是否领先选择披露。
'''
    (ROOT/'PROTOCOL.md').write_text(protocol,encoding='utf-8')
    print(json.dumps(dict(status='passed',batch_size=chosen,results=results),indent=2))


if __name__=='__main__':run()
