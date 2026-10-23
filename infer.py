"""Portable inference for the archived two-phase SeisFormer checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch
from seisformer.model import ModelConfig,SeisFormer
from seisformer.standardized import standardize

ROOT=Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load_model(checkpoint,device):
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    m=SeisFormer(ModelConfig(**c['model_config']))
    m.load_state_dict(c['model'],strict=True)
    return m.to(device).eval()


def load_windows(path,standardized=False,sample_rate=None,order=None):
    if path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as z:
            if 'x' not in z:raise ValueError('NPZ needs x and optional valid_mask')
            x=z['x'];mask=z['valid_mask'].astype(bool) if 'valid_mask' in z else None
    else:x=np.load(path,allow_pickle=False);mask=None
    if x.ndim==2:
        x=x[None]
        if mask is not None:mask=mask[None]
    if x.ndim!=3 or not len(x):raise ValueError('Input must be CxT or BxCxT')
    if mask is not None and mask.shape!=x.shape:raise ValueError('Mask shape differs from x')
    if standardized:
        if x.shape[1:]!=(3,6000):raise ValueError('Standardized windows must be Bx3x6000 at100Hz ENZ')
        if mask is None:mask=np.ones_like(x,dtype=bool)
        if not np.isfinite(x[mask]).all():raise ValueError('Nonfinite samples inside measured support')
        return np.where(mask,x,0).astype(np.float32),mask
    if sample_rate is None or order is None:raise ValueError('Raw NPY/NPZ requires --sample-rate and --order; or use --standardized')
    waves=[];supports=[]
    for i,record in enumerate(x):
        record=record.astype(np.float64)
        if mask is not None:record=np.where(mask[i],record,np.nan)
        encoded,support,_,_,_=standardize(record,sample_rate,order.upper())
        if encoded.shape!=(3,6000):raise ValueError('Raw input must become60s/6000samples after resampling; use mseed mode for long records')
        waves.append(encoded.astype(np.float32));supports.append(support)
    return np.stack(waves),np.stack(supports)


@torch.inference_mode()
def infer_windows(model,x,mask,device,event_threshold=.5,phase_threshold=.3,calibration=None,batch_size=16):
    result=[]
    for start in range(0,len(x),batch_size):
        xb=torch.from_numpy(x[start:start+batch_size]).to(device)
        mb=torch.from_numpy(mask[start:start+batch_size]).to(device)
        o=model(xb)
        prob=o['phase_logits'].float().sigmoid().masked_fill(~mb.any(1)[:,None],-1)
        peak,pick=prob.max(-1);event=o['event_logit'].float().sigmoid()
        for i in range(len(xb)):
            supported=bool(mb[i].any());ep=float(event[i]);accepted=supported and ep>=event_threshold
            row=dict(index=start+i,record_valid=supported,event_probability=ep,event_accepted=accepted,phases={})
            for j,p in enumerate(('P','S')):
                time=float(pick[i,j])/100 if supported else None
                confidence=max(0.,float(peak[i,j]))
                use=accepted and confidence>=phase_threshold
                entry=dict(candidate_time_s=time,peak_probability=confidence,accepted=use,time_s=time if use else None)
                if use and calibration is not None:
                    radius=calibration[p]['radius_s']
                    entry['historical_marginal_interval_s']=[max(0.,time-radius),min(59.99,time+radius)]
                row['phases'][p]=entry
            result.append(row)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'weights/best_detection.pt')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--standardized',action='store_true');p.add_argument('--sample-rate',type=float)
    p.add_argument('--order');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--event-threshold',type=float,default=.5);p.add_argument('--phase-threshold',type=float,default=.3)
    p.add_argument('--min-support',type=int,default=3);p.add_argument('--no-intervals',action='store_true')
    a=p.parse_args();torch.set_num_threads(a.threads)
    if a.batch_size<1 or a.min_support<1:raise ValueError('Positive batch/support required')
    model=load_model(a.checkpoint,a.device);fingerprint=digest(a.checkpoint)
    calibration=None
    if not a.no_intervals:
        calibration=json.loads((ROOT/'config/calibration.json').read_text(encoding='utf-8'))
        if calibration['checkpoint_sha256']!=fingerprint:
            raise ValueError('Calibration belongs to primary checkpoint; alternative weights require --no-intervals')
    result=dict(checkpoint_sha256=fingerprint,sample_rate_hz=100,component_order='ENZ',
        event_threshold=a.event_threshold,phase_threshold=a.phase_threshold)
    if a.input.suffix.lower() in ('.mseed','.ms','.miniseed','.sac'):
        from runtime.external_decoder import prepare,infer_events
        x,mask=prepare(a.input)
        events,ep,n=infer_events(model,x,mask,a.device,min_support=a.min_support,min_peak=a.phase_threshold,fallback=False)
        accepted=ep>=a.event_threshold
        result.update(mode='label_free_sliding_windows',duration_s=x.shape[1]/100,event_probability=ep,
            event_accepted=accepted,candidates=events,emitted_events=events if accepted else [],
            raw_candidate_clusters=n,min_support=a.min_support,fallback=False,
            scope='Waveform-only sliding inference. No catalogue/true pick selects candidates. Historical challenge scores used a different truth-assisted matching step.')
    else:
        x,mask=load_windows(a.input,a.standardized,a.sample_rate,a.order)
        result.update(mode='fixed_60s_windows',predictions=infer_windows(model,x,mask,a.device,
            a.event_threshold,a.phase_threshold,calibration,a.batch_size),
            interval_scope='Historical marginal CAL radius, not a calibrated conditional or new-domain coverage guarantee.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(a.output.resolve())


if __name__=='__main__':main()
