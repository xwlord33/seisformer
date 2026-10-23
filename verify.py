"""Verify archive identity and recompute historical metrics from saved predictions."""
from pathlib import Path
import hashlib
import json
import platform
import math

import numpy as np
import torch
from seisformer.metrics import summarize
from seisformer.model import ModelConfig,SeisFormer

ROOT=Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def compare(a,b,path='overall'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Metric keys differ: '+path)
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif a is None or isinstance(a,(str,bool)):
        if a!=b:raise ValueError('Metric differs: '+path)
    elif not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError(f'Metric differs: {path}: {a} vs {b}')


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        p=ROOT/entry['path']
        if p.stat().st_size!=entry['bytes'] or digest(p)!=entry['sha256']:
            raise ValueError('Archive file changed: '+entry['path'])
    rows={}
    for role in ('cal','test'):
        with np.load(ROOT/f'evidence/internal/{role}_predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        original=json.loads((ROOT/f'evidence/internal/{role}_metrics.json').read_text(encoding='utf-8'))
        mask=a['event_mask'];computed=summarize({k:v[mask] for k,v in a.items()})
        compare(computed,original['overall'])
        if len(mask)!=original['rows'] or int((~mask).sum())!=original['excluded_unknown_event_crops']:
            raise ValueError('Window eligibility accounting differs')
        for source,v in original['by_source'].items():
            use=mask&(a['source']==source)
            compare(summarize({k:x[use] for k,x in a.items()}),v,'by_source.'+source)
        rows[role]=dict(rows=len(mask),evaluated=int(mask.sum()),metrics_match=True,
            P_mae_s=computed['P']['mae_s'],S_mae_s=computed['S']['mae_s'],detection_f1=computed['detection']['0.5']['f1'])
    ck=torch.load(ROOT/'weights/best_detection.pt',map_location='cpu',weights_only=False)
    model=SeisFormer(ModelConfig(**ck['model_config']));model.load_state_dict(ck['model'],strict=True)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=428774 or ck['epoch']!=16:raise ValueError('Unexpected primary model identity')
    cal=json.loads((ROOT/'config/calibration.json').read_text(encoding='utf-8'))
    if cal['checkpoint_sha256']!=digest(ROOT/'weights/best_detection.pt'):raise ValueError('Calibration identity differs')
    report=dict(status='passed',files_verified=len(manifest['files']),metrics=rows,parameters=parameters,
        selected_epoch_human=17,checkpoint_strict_load=True,calibration_identity=True,
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        source_test_waveforms_read=False,training_performed=False,
        scope='File integrity, exact cached-protocol metrics, and model load; not new test inference or a first-tier claim.')
    p=ROOT/'validation/verification.json';p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
