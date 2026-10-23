"""Export verified comparison evidence and inference-only states for local review."""
from pathlib import Path
import gzip
import json
import shutil

import torch

from scripts.recent_dual_core import PROJECT,read,sha
from scripts.rebuild_standardized import atomic_json


def run():
    experiment=PROJECT/'outputs/recent_dual_comparison_v1';paper=PROJECT/'outputs/paper_ps'
    result=read(experiment/'RESULTS.json');assert result['status']=='complete' and len(result['runs'])==6
    target=paper/'reproducibility/recent_dual';target.mkdir(exist_ok=True)
    for name in ('RESULTS.json','RESULTS.md','runtime.json','environment.json','PROTOCOL.md',
                 'PROTOCOL_precision_addendum.md','config.json','config_fp32_lstm.json','ACTIVE_CONFIG.json',
                 'selected_learning_rates.json','gpu_preflight.json','preparation.json'):
        shutil.copy2(experiment/name,target/name)
    records=[]
    for item in result['runs']:
        original=Path(item['directory']);folder=target/'runs'/original.name;folder.mkdir(parents=True,exist_ok=True)
        for name in ('completion.json','run.json','history.jsonl','test_metrics.json','validation_metrics.json',
                     'test_predictions.npz','validation_predictions.npz'):
            shutil.copy2(original/name,folder/name)
        checkpoint=torch.load(original/'best.pt',map_location='cpu',weights_only=False)
        export=dict(format_version='recent-dual-inference-v1',architecture=item['architecture'],
            model=checkpoint['model'],seed=item['seed'],selected_epoch=checkpoint['epoch'],
            original_training_checkpoint_sha256=sha(original/'best.pt'),
            canonical_configuration='config_fp32_lstm.json',input_samples=6000,sample_rate_hz=100,archive_component_order='ENZ')
        weights=folder/'inference.pt';torch.save(export,weights)
        again=torch.load(weights,map_location='cpu',weights_only=True)
        assert set(again['model'])==set(checkpoint['model'])
        assert all(torch.equal(v,again['model'][k]) for k,v in checkpoint['model'].items())
        records.append(dict(architecture=item['architecture'],seed=item['seed'],path=weights.relative_to(target).as_posix(),
            inference_sha256=sha(weights),training_checkpoint_sha256=export['original_training_checkpoint_sha256'],all_state_tensors_identical=True))
    vendor=target/'upstream';vendor.mkdir(exist_ok=True)
    for name in ('cross_twostreameqt.py','LICENSE','PROVENANCE.json','stead.json.v0','config_stead.json'):
        shutil.copy2(PROJECT/'third_party/twostream_eqt_2025'/name,vendor/name)
    sources=read(experiment/'config_fp32_lstm.json')['source_sha256']
    for name,value in sources.items():
        source=Path(name);assert sha(source)==value
        copied=target/'source_snapshot'/source.relative_to(PROJECT)
        copied.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,copied)
    for name in ('precision_probe.json','recovery_preflight.json','full_fp32_equivalence.json'):
        folder=target/'numerical_compatibility';folder.mkdir(exist_ok=True);shutil.copy2(experiment/'nan_diagnosis'/name,folder/name)
    with gzip.open(target/'matched_train_val_rows.json.gz','wb') as stream:
        stream.write(Path(read(experiment/'config_fp32_lstm.json')['train_val_rows']).read_bytes())
    atomic_json(target/'inference_state_manifest.json',dict(status='verified',states=records,result_sha256=sha(experiment/'RESULTS.json')))
    (target/'README.md').write_text('''# Matched recent dual-domain comparison

This folder preserves all six completed 20-epoch runs, selected inference states, exact test/validation predictions, learning-rate selection, runtime and protocol. The original 1M historical model remains in the adjacent main-model package.

`runs/*/inference.pt` contains the selected model state without optimizer history. Every tensor is verified identical to its original selected training checkpoint; both hashes are recorded in `inference_state_manifest.json`. Construct the corresponding architecture with `scripts.recent_dual_core.build_model(architecture)` from the complete project, then load `checkpoint['model']` strictly. Use the archive's 100-Hz ENZ inputs, masks and adapter documented in the protocol. Input preprocessing is part of the model adapter for TwoStream-EQT.

`source_snapshot/` contains exact experiment dependencies for inspection, and `upstream/` preserves the unchanged reference network and GPLv3 license. Existing project scripts and provider-controlled waveforms are required to repeat training; this is a local source/evidence package rather than a self-contained waveform distribution. `matched_train_val_rows.json.gz` preserves the exact comparison rows; the common test list is `../row_manifests/test_entries.json.gz`.

`config_fp32_lstm.json` is the active amended configuration. `config.json` records the original protocol. The reference recurrent operations use FP32 within mixed precision, and final evaluation/runtime use FP32 with TF32 disabled. Compatibility evidence is retained in `numerical_compatibility/`; incomplete BF16-only attempts are excluded from formal results and remain in the original experiment directory.
''',encoding='utf-8')
    print(json.dumps(dict(status='exported',verified_inference_states=len(records),folder=str(target)),ensure_ascii=False,indent=2))


if __name__=='__main__':run()
