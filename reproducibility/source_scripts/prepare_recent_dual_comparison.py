"""Prepare exact matched rows and verify the fast reader against the original."""
from pathlib import Path
import gzip
import json

import numpy as np
import torch

from scripts.recent_dual_core import make_cache,MatchedWaveforms,read,sha,PROJECT
from scripts.rebuild_standardized import atomic_json
from seisformer.standardized import StandardizedWaveforms

ROOT=PROJECT/'outputs/recent_dual_comparison_v1'


def run():
    ROOT.mkdir(exist_ok=True)
    rows_path=PROJECT/'outputs/standardized_dual_262k/selected_rows.json'
    snapshot=PROJECT/'outputs/standardized_dual_262k/snapshot.json'
    rows=read(rows_path)
    assert len(rows['train'])==262144 and len(rows['val'])==16384
    atomic_json(ROOT/'preparation.json',dict(status='building_train_val_cache',training_started=False))
    records={}
    for split in ('train','val'):
        records[split]=make_cache(rows[split],'F:/data_re',snapshot,split,ROOT/'cache'/split)
        atomic_json(ROOT/'preparation.json',dict(status='cache_'+split+'_complete',records=records,training_started=False))
    # Identical evaluation crops to the manuscript reader, plus multiple epochs
    # of deterministic pairwise train crops independent of model RNG or workers.
    parity=[]
    for split in ('train','val'):
        cache=MatchedWaveforms(ROOT/'cache'/split)
        indices=np.random.default_rng(42).choice(len(cache),64,replace=False)
        only={}
        chosen=[rows[split][int(i)] for i in indices]
        for p,r in chosen:only.setdefault(p,set()).add(int(r))
        original=StandardizedWaveforms('F:/data_re',split,seed=42,random_crop=False,snapshot=snapshot,only_rows=only)
        original.entries=[tuple(v) for v in chosen]
        for position,i in enumerate(indices):
            a,b=original[position],cache[int(i)]
            for key in ('x','time_mask','valid_mask','picks','phase_mask','event','event_mask'):
                assert torch.equal(a[key],b[key]),(split,int(i),key)
            assert a['source']==b['source'] and a['event_group']==b['event_group']
            for epoch in (0,1,19):
                first,second=cache[(int(i),epoch)],cache[(int(i),epoch)]
                assert all(torch.equal(first[k],second[k]) for k in ('x','picks','time_mask','event_mask'))
        parity.append(dict(split=split,records=64,exact_original_fields=True,paired_epoch_crop_checks=True))
        del original,cache
    test_gz=PROJECT/'outputs/paper_ps/reproducibility/row_manifests/test_entries.json.gz'
    with gzip.open(test_gz,'rt',encoding='utf-8') as stream:test=json.load(stream)
    assert len(test)==127940
    atomic_json(ROOT/'test_rows.json',test)
    config=dict(version='recent-dual-matched-ps-v1',created_date='2026-10-22',data='F:/data_re',
        snapshot=str(snapshot),test_snapshot=str(snapshot),train_val_rows=str(rows_path),test_rows=str(ROOT/'test_rows.json'),
        architectures=['seisformer','twostream_eqt'],seeds=[42,43,44],data_seed=42,
        train_records=262144,validation_records=16384,test_records=127940,
        tune_records=32768,tune_validation_records=4096,tune_epochs=3,
        learning_rates=[1e-4,3e-4,1e-3],epochs=20,batch_size=None,workers=4,threads=4,
        optimizer='AdamW',weight_decay=.01,lr_schedule='cosine',minimum_lr_factor=.1,
        clip_norm=1.,precision='BF16 training/selection; FP32 final evaluation and runtime',
        initialization='All tuning and final models initialized independently from scratch; no pretrained weights.',
        phase_supervision='Same sigma=10 samples, peak weight=60, unknown/time masks, phase weight=2.',
        event_supervision='SeisFormer: native window event BCE. TwoStream-EQT: native dense P-to-P+1.4(S-P) BCE; incomplete pairs masked.',
        native_preprocessing='TwoStream-EQT public annotate_batch_pre: ENZ to ZNE, component demeaning, shared std, six-sample cosine taper; upstream network unchanged.',
        selection='Minimum unthresholded mean P/S MAE on the fixed validation set, at epoch boundaries.',
        tuning='Same three learning-rate candidates, same train/validation subsets and 3-epoch budget for each architecture; no test access.',
        metrics='Unconditional P/S MAE; picking F1 within 0.3s. Fixed thresholds plus validation-selected thresholds at 1% noise budget.',
        runtime='Same RTX 5070 Ti and CPU 2 threads; FP32 B1/B64, STFT and native preprocessor included; disk/transfer excluded.',
        reference='Historical 1M best model is context only and never used to rank the matched 262k models.',
        cache_records=records,reader_parity=parity,
        input_sha256={str(p):sha(p) for p in (rows_path,snapshot,test_gz,ROOT/'test_rows.json')})
    atomic_json(ROOT/'config.preflight.json',config)
    atomic_json(ROOT/'preparation.json',dict(status='ready_for_gpu_preflight',reader_parity=parity,training_started=False))
    print(json.dumps(dict(status='prepared',reader_parity=parity,cache_GiB={k:round(sum(v['bytes'].values())/2**30,2) for k,v in records.items()}),indent=2))


if __name__=='__main__':run()
