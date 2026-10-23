"""Collect same-task P/S evidence and recompute primary-checkpoint calibration."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import numpy as np
from seisformer.metrics import summarize,conformal_radius

ROOT=Path('outputs/paper_ps')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def run():
    output=ROOT/'analysis';output.mkdir(exist_ok=True)
    used={}
    def get(p):used[str(p)]=sha(p);return read(p)
    def npz(p):used[str(p)]=sha(p);return arrays(p)
    base=Path('outputs/standardized_dual_1m_20ep')
    report=get(base/'eval_test_det_cal/test_metrics.json')
    cal=npz(base/'eval_cal_det/cal_predictions.npz');test=npz(base/'eval_test_det/test_predictions.npz')
    recomputed=summarize({k:v[test['event_mask']] for k,v in test.items()})
    if recomputed!=report['overall']:raise ValueError('Saved report differs from predictions')
    runs={}
    names=['standardized_time_ondual_262k','standardized_time_seed43','standardized_time_seed44',
        'standardized_dual_262k','standardized_dual_seed43','standardized_dual_seed44b',
        'standardized_dual_expanded_524k','standardized_dual_1m_10ep','standardized_dual_1m_20ep',
        'standardized_dual_1m_seed43','standardized_dual_1m_seed44','standardized_dual_ppaa_524k',
        'standardized_dual_1m_ppaa','standardized_unet_524k','standardized_phasenet_524k']
    for name in names:
        root=Path('outputs')/name;item=dict(run=name)
        for file in ('run.json','progress.json','best_validation.json','best_detection_validation.json'):
            p=root/file
            if p.exists():item[file[:-5]]=get(p)
        for folder in ('eval_test','eval_test_det'):
            p=root/folder/'test_metrics.json'
            if p.exists():item[folder]=get(p)
        runs[name]=item
    controlled=[]
    for seed,time,dual in ((42,names[0],names[3]),(43,names[1],names[4]),(44,names[2],names[5])):
        row=dict(seed=seed)
        for kind,name in (('time',time),('dual',dual)):
            metric=runs[name]['eval_test']['overall']
            row[kind]={p:metric[p]['mae_s'] for p in ('P','S')}
        row['difference']={p:row['dual'][p]-row['time'][p] for p in ('P','S')};controlled.append(row)
    seedstats={kind:{p:dict(mean=float(np.mean([r[kind][p] for r in controlled])),
        sd=float(np.std([r[kind][p] for r in controlled],ddof=1))) for p in ('P','S')} for kind in ('time','dual','difference')}
    bootstrap={n:get(Path('outputs')/(n+'.json')) for n in ('bootstrap_dual_vs_time','bootstrap_1m_vs_524k',
        'bootstrap_ppaa_vs_dual','bootstrap_1mppaa_vs_1mdet','per_source_bh_dual_vs_time')}
    snapshot=get('outputs/standardized_expanded_full/snapshot.json');counts=defaultdict(Counter)
    for key,n in snapshot['composition'].items():
        source,role=key.rsplit('/',1);counts[source][role]+=n
    total=Counter()
    for c in counts.values():total.update(c)
    coverage=[];budget=[];rng=np.random.default_rng(42)
    for j,phase in enumerate(('P','S')):
        known=cal['truth'][:,j]>=0
        global_errors=np.abs(cal['pred'][known,j]-cal['truth'][known,j])/100
        radius=conformal_radius(global_errors,.1)
        for source in sorted(set(test['source'])):
            cm=(cal['source']==source)&(cal['truth'][:,j]>=0)
            tm=(test['source']==source)&(test['truth'][:,j]>=0)
            if not tm.any():continue
            ce=np.abs(cal['pred'][cm,j]-cal['truth'][cm,j])/100
            te=np.abs(test['pred'][tm,j]-test['truth'][tm,j])/100
            if len(ce)<20:continue
            local=conformal_radius(ce,.1)
            coverage.append(dict(source=str(source),phase=phase,n_cal=len(ce),n_test=len(te),
                cal_groups=len(np.unique(cal['group'][cm])),test_groups=len(np.unique(test['group'][tm])),
                global_radius=radius,local_radius=local,global_coverage=float((te<=radius).mean()),
                local_coverage=float((te<=local).mean())))
            # Supplementary sensitivity analysis only: no universal label budget
            # or new-network conditional coverage theorem is inferred.
            others=known&(cal['source']!=source)
            prior=np.abs(cal['pred'][others,j]-cal['truth'][others,j])/100
            for n in (0,25,100,250,500,1000):
                if n>len(ce):continue
                cover=[]
                for draw in range(30):
                    pool=rng.choice(prior,200,replace=False)
                    if n:pool=np.r_[pool,rng.choice(ce,n,replace=False)]
                    rad=conformal_radius(pool,.1);cover.append(float((te<=rad).mean()))
                budget.append(dict(source=str(source),phase=phase,n_local=n,mean_coverage=float(np.mean(cover)),
                    q10=float(np.quantile(cover,.1)),q90=float(np.quantile(cover,.9))))
    calibration_summary={}
    for mode in ('global','local'):
        deviations=np.abs([r[mode+'_coverage']-.9 for r in coverage])
        calibration_summary[mode]=dict(mean_absolute_deviation=float(np.mean(deviations)),
            maximum_absolute_deviation=float(np.max(deviations)),
            minimum_coverage=min(r[mode+'_coverage'] for r in coverage),maximum_coverage=max(r[mode+'_coverage'] for r in coverage))
    selective=[]
    for j,p in enumerate(('P','S')):
        known=test['truth'][:,j]>=0;e=np.abs(test['pred'][known,j]-test['truth'][known,j])/100
        conf=test['confidence'][known,j];order=np.argsort(-conf,kind='stable')
        for fraction in (1.,.95,.9,.8):
            n=int(np.ceil(len(e)*fraction));use=order[:n]
            selective.append(dict(phase=p,fraction=fraction,n=n,mae_s=float(e[use].mean()),
                hit_03=float((e[use]<=.3).mean())))
    output_data=dict(primary=report,primary_predicted_metrics=recomputed,runs=runs,controlled_seeds=controlled,
        controlled_summary=seedstats,bootstrap=bootstrap,snapshot_totals=dict(total),
        source_inventory={s:dict(c) for s,c in counts.items()},
        primary_train_composition=get(base/'selected_composition.json'),
        primary_source_calibration=coverage,calibration_summary=calibration_summary,
        calibration_budget=budget,selective_primary=selective,
        evidence_sha256=used,scope='Original P/S checkpoints only. No six-phase trained candidate is attributed to the P/S model. All new calibration statistics recomputed from same primary checkpoint CAL/TEST caches.')
    (output/'submission_evidence.json').write_text(json.dumps(output_data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(controlled_summary=seedstats,snapshot_totals=dict(total),calibration_summary=calibration_summary,
        source_coverage=coverage),indent=2))


if __name__=='__main__':run()
