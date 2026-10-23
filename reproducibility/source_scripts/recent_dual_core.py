"""Matched P/S experiment utilities for the published TwoStream-EQT network.

Only this adapter knows the upstream dense detection output convention. The
upstream network source is kept byte-for-byte unchanged and hash checked.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from seisformer.model import ModelConfig, SeisFormer
from seisformer.metrics import summarize, binary_scores
from scripts.rebuild_standardized import atomic_json

PROJECT=Path(__file__).resolve().parents[1]
UPSTREAM=PROJECT/'third_party/twostream_eqt_2025'
SOURCE_BLOB='85afad0d97bd2216d5b9af1fd2ade88b276ff9c2'
VERSION='recent-dual-matched-ps-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def native_class():
    path=UPSTREAM/'cross_twostreameqt.py';data=path.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==SOURCE_BLOB
    name='seisbench.models.recent_dual_crosstwostreameqt'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].CrossTwoStreamEQT


class TwoStreamAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.native=native_class()(cnn_blocks=7,res_cnn_blocks=7,lstm_blocks=3,
            drop_rate=.3,original_compatible=False,sampling_rate=100,norm='std',
            eqt_mode='twostream',logits=True,component_order='ZNE')
        # The released recurrent path develops nonfinite outputs under cuDNN
        # BF16 on this runtime. Execute the unchanged nn.LSTM math in FP32;
        # state names, parameters and full-FP32 inference are unchanged.
        self.fp32_recurrent_modules=fp32_recurrent(self.native)

    def forward(self,x,time_mask=None):
        # Public model metadata specifies ZNE; the shared archive stores ENZ.
        zne=x[:,[2,1,0]].contiguous()
        zne=self.native.annotate_batch_pre(zne,{})
        event,p,s=self.native(zne)
        valid=torch.ones_like(event,dtype=torch.bool) if time_mask is None else time_mask
        pooled=event.masked_fill(~valid,-1e4).amax(-1)
        return dict(phase_logits=torch.stack((p,s),1),event_logit=pooled,dense_event_logits=event)


def fp32_recurrent(model):
    names=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.LSTM):
            original=module.forward
            def stable(*args,_original=original,**kwargs):
                def cast(value):
                    if torch.is_tensor(value) and value.is_floating_point():return value.float()
                    if isinstance(value,tuple):return tuple(cast(v) for v in value)
                    return value
                device=args[0].device.type
                with torch.autocast(device,enabled=False):return _original(*(cast(v) for v in args),**{k:cast(v) for k,v in kwargs.items()})
            module.forward=stable;names.append(name)
    return names


def build_model(architecture):
    if architecture=='seisformer':return SeisFormer(ModelConfig(wavelet=True,stft=True))
    if architecture=='twostream_eqt':return TwoStreamAdapter()
    raise ValueError(architecture)


def forward(model,architecture,batch,device='cuda'):
    x=batch['x'].to(device,non_blocking=True)
    return model(x,batch['time_mask'].to(device,non_blocking=True)) if architecture=='twostream_eqt' else model(x)


def matched_loss(out,batch):
    """Identical Gaussian P/S supervision; each native detection head is retained.

    For the upstream dense detector, use its conventional P--S event interval,
    extending to P+1.4(S-P). Incomplete P/S pairs are not labelled as silence.
    This is a matched-data adaptation, not the authors' 100-epoch STEAD recipe.
    """
    logits=out['phase_logits'].float();device=logits.device
    p=batch['picks'].to(device);valid=batch['time_mask'].to(device)
    mask=valid[:,None]&batch['phase_mask'].to(device)[:,:,None]
    axis=torch.arange(logits.shape[-1],device=device)
    target=torch.exp(-.5*((axis[None,None]-p[...,None])/10.)**2)*(p>=0)[...,None]
    phase=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*(1+60*target)*mask).sum()/mask.sum().clamp_min(1)
    if 'dense_event_logits' in out:
        known_pair=(p>=0).all(1)&(p[:,1]>=p[:,0])
        noise=batch['event'].to(device)<.5
        det_mask=valid&(known_pair|noise)[:,None]
        end=p[:,0]+1.4*(p[:,1]-p[:,0])
        event_target=((axis[None]>=p[:,0,None])&(axis[None]<=end[:,None])&known_pair[:,None]).float()
        event=(F.binary_cross_entropy_with_logits(out['dense_event_logits'].float(),event_target,reduction='none')*det_mask).sum()/det_mask.sum().clamp_min(1)
    else:
        em=batch['event_mask'].to(device)
        event=(F.binary_cross_entropy_with_logits(out['event_logit'].float(),batch['event'].to(device),reduction='none')*em).sum()/em.sum().clamp_min(1)
    return 2*phase+event


def make_cache(entries,data_root,snapshot_path,split,output):
    """Copy selected variable-length records losslessly into two sequential files."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    entry_hash=hashlib.sha256(json.dumps(entries,separators=(',',':')).encode()).hexdigest()
    marker=output/'COMPLETE.json'
    if marker.exists():
        record=read(marker)
        assert record['entries_sha256']==entry_hash and record['split']==split
        assert record.get('picks_dtype')=='float32','Cache must retain fractional resampled arrivals.'
        assert all((output/k).stat().st_size==v for k,v in record['bytes'].items())
        return record
    snapshot=read(snapshot_path);by_path={}
    assert len({(str(p),int(r)) for p,r in entries})==len(entries)
    for i,(path,row) in enumerate(entries):by_path.setdefault(str(path),[]).append((i,int(row)))
    n=len(entries);length=np.zeros(n,np.int64);picks=np.zeros((n,2),np.float32);noise=np.zeros(n,bool)
    sources=['']*n;groups=['']*n;identity=['']*n;frozen=[]
    root=Path(data_root).resolve()
    for path,items in by_path.items():
        directory=Path(path);relative=directory.resolve().relative_to(root).as_posix()
        assert relative in snapshot['completion_sha256']
        assert sha(directory/'COMPLETE.json')==snapshot['completion_sha256'][relative]
        frozen.append((str(directory/'COMPLETE.json'),snapshot['completion_sha256'][relative]))
        with np.load(directory/'index.npz',allow_pickle=False) as z:
            index={key:z[key] for key in ('split','length','picks','noise','source','group')}
            for i,row in items:
                assert str(index['split'][row])==split,(path,row,split)
                length[i]=index['length'][row];picks[i]=index['picks'][row];noise[i]=index['noise'][row]
                sources[i]=str(index['source'][row]);groups[i]=str(index['group'][row]);identity[i]=f'{relative}:{row}'
    wave_offset=np.r_[0,np.cumsum(3*length)];mask_offset=np.r_[0,np.cumsum((3*length+7)//8)]
    waves=np.memmap(output/'waves.f16',mode='w+',dtype=np.float16,shape=(int(wave_offset[-1]),))
    masks=np.memmap(output/'masks.u8',mode='w+',dtype=np.uint8,shape=(int(mask_offset[-1]),))
    for path,items in by_path.items():
        directory=Path(path)
        raw=np.memmap(directory/'waveforms.f16',mode='r',dtype=np.float16)
        bits=np.memmap(directory/'valid.bits',mode='r',dtype=np.uint8)
        with np.load(directory/'index.npz',allow_pickle=False) as z:
            offsets=z['offset'];mask_offsets=z['mask_offset']
            for i,row in items:
                off=int(offsets[row]);mo=int(mask_offsets[row])
                waves[wave_offset[i]:wave_offset[i+1]]=raw[off:off+3*length[i]]
                masks[mask_offset[i]:mask_offset[i+1]]=bits[mo:mo+(3*length[i]+7)//8]
    waves.flush();masks.flush();del waves,masks
    np.savez_compressed(output/'metadata.npz',length=length,wave_offset=wave_offset,mask_offset=mask_offset,
        picks=picks,noise=noise,source=np.asarray(sources),group=np.asarray(groups),identity=np.asarray(identity))
    record=dict(format_version=VERSION,entries_sha256=entry_hash,split=split,rows=n,
        source_completion_hashes=frozen,metadata_sha256=sha(output/'metadata.npz'),
        bytes={k:(output/k).stat().st_size for k in ('waves.f16','masks.u8','metadata.npz')},
        lossless=True,picks_dtype='float32',physical_rate_hz=100,component_order='ENZ')
    atomic_json(marker,record);return record


class MatchedWaveforms(Dataset):
    def __init__(self,cache,limit=0,data_seed=42):
        self.cache=Path(cache);self.record=read(self.cache/'COMPLETE.json')
        assert sha(self.cache/'metadata.npz')==self.record['metadata_sha256']
        with np.load(self.cache/'metadata.npz',allow_pickle=False) as z:self.meta={k:z[k] for k in z.files}
        self.n=min(limit,self.record['rows']) if limit else self.record['rows']
        self.data_seed=data_seed;self.waves=self.masks=None

    def __len__(self):return self.n

    def __getstate__(self):
        value=self.__dict__.copy();value['waves']=value['masks']=None;return value

    def __getitem__(self,item):
        i,epoch=item if isinstance(item,tuple) else (item,None)
        if self.waves is None:
            self.waves=np.memmap(self.cache/'waves.f16',mode='r',dtype=np.float16)
            self.masks=np.memmap(self.cache/'masks.u8',mode='r',dtype=np.uint8)
        m=self.meta;n=int(m['length'][i]);size=min(n,6000)
        identity=f"{self.data_seed}:{m['identity'][i]}"
        if epoch is not None:identity+=f':epoch={epoch}'
        rng=np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest()[:16],16))
        start=int(rng.integers(max(0,n-6000)+1))
        raw=self.waves[m['wave_offset'][i]:m['wave_offset'][i+1]].reshape(3,n)
        mask=np.unpackbits(self.masks[m['mask_offset'][i]:m['mask_offset'][i+1]],count=3*n).reshape(3,n).astype(bool)
        x=np.zeros((3,6000),np.float32);valid=np.zeros((3,6000),bool)
        x[:,:size]=raw[:,start:start+size];valid[:,:size]=mask[:,start:start+size]
        picks=m['picks'][i].copy()-start;known=(picks>=0)&(picks<size)
        for j in range(2):
            if known[j] and not valid[:,int(picks[j])].any():known[j]=False
        picks[~known]=-1;noise=bool(m['noise'][i])
        return dict(x=torch.from_numpy(x),time_mask=torch.from_numpy(valid.any(0)),
            valid_mask=torch.from_numpy(valid),picks=torch.from_numpy(picks),phase_mask=torch.from_numpy(known|noise),
            event=torch.tensor(float(not noise)),event_mask=torch.tensor(bool(noise or known.any())),
            source=str(m['source'][i]),event_group=str(m['group'][i]),row=i,crop_start=start)


class PairedEpochSampler(Sampler):
    def __init__(self,n,seed):self.n=n;self.seed=seed;self.epoch=0
    def __len__(self):return self.n
    def __iter__(self):
        order=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.seed*1000+self.epoch)).tolist()
        return iter((i,self.epoch) for i in order)


@torch.inference_mode()
def evaluate(model,architecture,loader,device='cuda',bf16=False):
    model.eval();arrays={k:[] for k in ('truth','pred','confidence','event','event_prob','source','event_mask','row','group','crop_start')}
    for batch in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16) if bf16 and device=='cuda' else nullcontext():
            result=forward(model,architecture,batch,device)
        prob=result['phase_logits'].float().sigmoid().masked_fill(~batch['time_mask'].to(device)[:,None],-1)
        confidence,pred=prob.max(-1)
        values=dict(truth=batch['picks'].numpy(),pred=pred.cpu().numpy(),confidence=confidence.cpu().numpy(),
            event=batch['event'].numpy(),event_prob=result['event_logit'].float().sigmoid().cpu().numpy(),
            source=np.asarray(batch['source']),event_mask=batch['event_mask'].numpy(),row=batch['row'].numpy(),
            group=np.asarray(batch['event_group']),crop_start=batch['crop_start'].numpy())
        for key,value in values.items():arrays[key].append(value)
    a={k:np.concatenate(v) for k,v in arrays.items()};eligible=a['event_mask']
    report=dict(overall=summarize({k:v[eligible] for k,v in a.items()}),by_source={},total_crops=len(eligible),
        excluded_unknown_event_crops=int((~eligible).sum()),precision='BF16-validation' if bf16 else 'FP32')
    for source in np.unique(a['source']):
        use=eligible&(a['source']==source)
        if use.any():report['by_source'][str(source)]=summarize({k:v[use] for k,v in a.items()})
    return report,a


def operating_point(validation,noise_budget=.01):
    """Choose an event threshold from known validation noise, phases by val F1."""
    a={k:v[validation['event_mask']] for k,v in validation.items()}
    noise=a['event']<.5;scores=np.sort(a['event_prob'][noise])[::-1];allowed=int(np.floor(noise_budget*len(scores)))
    threshold=float(np.nextafter(scores[allowed],np.float32(np.inf))) if allowed<len(scores) else 0.
    phase_thresholds=[]
    for j in range(2):
        choices=[]
        for value in (.05,.1,.2,.3,.5,.7,.9):
            result=phase_scores(a,j,threshold,value);choices.append((result['f1'],value))
        phase_thresholds.append(max(choices)[1])
    return dict(event_threshold=threshold,phase_thresholds=phase_thresholds,validation_noise_budget=noise_budget,
        validation_noise_n=int(noise.sum()),validation_allowed_fp=allowed)


def phase_scores(a,j,event_threshold,phase_threshold):
    known=a['truth'][:,j]>=0;noise=a['event']<.5
    emit=(a['event_prob']>=event_threshold)&(a['confidence'][:,j]>=phase_threshold)
    hit=known&emit&(np.abs(a['pred'][:,j]-a['truth'][:,j])<=30)
    tp=int(hit.sum());fp=int((emit&(known|noise)&~hit).sum());fn=int((known&~hit).sum())
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                noise_fpr=float(emit[noise].mean()) if noise.any() else None)


def calibrated_metrics(predictions,point):
    a={k:v[predictions['event_mask']] for k,v in predictions.items()}
    det=binary_scores(a['event']>.5,a['event_prob']>=point['event_threshold'])
    return dict(detection=det,noise_fpr=float((a['event_prob'][a['event']<.5]>=point['event_threshold']).mean()),
        **{phase:phase_scores(a,j,point['event_threshold'],point['phase_thresholds'][j]) for j,phase in enumerate(('P','S'))})
