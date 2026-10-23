"""Illustrate preserved test predictions on identified source collections."""
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np
from scripts.make_ps_submission_figures import BLUE, RED, save, plt
from scripts.recent_dual_core import MatchedWaveforms

ROOT=Path('outputs/paper_ps')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run():
    predictions=Path('outputs/standardized_dual_1m_20ep/eval_test_det/test_predictions.npz')
    with np.load(predictions,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    ds=MatchedWaveforms('outputs/recent_dual_comparison_v1/cache/test',data_seed=42)
    assert len(ds)==len(a['truth'])==127940
    assert np.array_equal(ds.meta['source'],a['source'])
    assert np.array_equal(ds.meta['group'],a['group'])
    with gzip.open(ROOT/'reproducibility/row_manifests/test_entries.json.gz','rt',encoding='utf-8') as f:
        entries=json.load(f)
    errors=np.abs(a['pred']-a['truth'])/100
    fig,axes=plt.subplots(3,2,figsize=(7.1,5.55),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    records=[];data={}
    for i,source in enumerate(('STEAD','TXED','PNW')):
        # All two-label crops with margins for both onset displays are eligible.
        eligible=(a['source']==source)&(a['truth'].min(1)>=400)&(a['truth'].max(1)<5400)
        pool=np.flatnonzero(eligible)
        order=np.lexsort((a['row'][pool],errors[pool].sum(1)))
        row=int(pool[order[len(pool)//2]])
        b=ds[row];x=b['x'].numpy();valid=b['valid_mask'].numpy()
        assert b['source']==source and b['event_group']==a['group'][row]
        assert np.array_equal(b['picks'].numpy(),a['truth'][row])
        truth=a['truth'][row]/100;pred=a['pred'][row]/100
        peak=max(float(np.percentile(np.abs(x[valid]),99.5)),1e-8)
        display=np.clip(x/peak,-1,1);time=np.arange(x.shape[-1])/100
        for j in range(3):
            axes[i,0].plot(time[::2],display[j,::2]+(2-j)*1.4,lw=.38,color=['#6f96ad','#80a69d','#656d78'][j])
        for j,color in enumerate((BLUE,RED)):
            axes[i,0].axvline(truth[j],color=color,ls='--',lw=.9,label=('Reference '+('P','S')[j]) if i==0 else None)
            axes[i,0].axvline(pred[j],color=color,lw=.9,label=('Predicted '+('P','S')[j]) if i==0 else None)
        axes[i,0].set_yticks([0,1.4,2.8],['Z','N','E']);axes[i,0].set_xlim(0,60)
        axes[i,0].set_title(f'({chr(97+i*2)}) {source}; test row {row}',loc='left',fontsize=8)
        local=(time>=truth[1]-4)&(time<=truth[1]+6)
        for j in range(3):axes[i,1].plot(time[local]-truth[1],display[j,local]+(2-j)*1.4,lw=.5,color=['#6f96ad','#80a69d','#656d78'][j])
        axes[i,1].axvline(0,color=RED,ls='--',lw=.9)
        axes[i,1].axvline(pred[1]-truth[1],color=RED,lw=.9)
        axes[i,1].set_yticks([]);axes[i,1].set_xlim(-4,6)
        axes[i,1].set_title(f'({chr(98+i*2)}) |P/S error| = {errors[row,0]:.2f}/{errors[row,1]:.2f} s',loc='left',fontsize=8)
        records.append(dict(source=source,test_row=row,event_group=str(a['group'][row]),
            shard=entries[row][0],shard_row=entries[row][1],crop_start=int(b['crop_start']),
            reference_s=truth.tolist(),prediction_s=pred.tolist(),errors_s=errors[row].tolist(),
            eligible_count=len(pool),selection='Median summed P/S absolute error among source test crops with both labels between 4 and 54 seconds; ties by test row',
            split='test',reader_matches_saved_truth=True))
        data[f'x_{i}']=x;data[f'valid_{i}']=valid;data[f'truth_{i}']=truth;data[f'pred_{i}']=pred
    axes[0,0].legend(frameon=False,ncol=4,fontsize=5.7,loc='upper right',handlelength=1.4,columnspacing=.7)
    axes[-1,0].set_xlabel('Time in test crop (s)');axes[-1,1].set_xlabel('Time relative to reference S (s)')
    save(fig,'ps_heldout_waveforms')
    np.savez_compressed(ROOT/'analysis/heldout_waveform_data.npz',**data)
    result=dict(prediction_file=str(predictions),prediction_sha256=sha(predictions),
        primary_checkpoint_sha256='08afa0d899bbc03e5680dff2b49768aacb1ce5fdfe07525d037b1c87e9ff3342',
        test_records=127940,source_collections=['STEAD','TXED','PNW'],examples=records,
        source_data='Existing frozen project test partition and unchanged primary predictions',
        new_training=False,new_model_evaluation=False)
    (ROOT/'analysis/heldout_waveform_selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(examples=[dict(source=r['source'],row=r['test_row'],errors_s=r['errors_s']) for r in records],prediction_sha256=result['prediction_sha256']),indent=2))


if __name__=='__main__':run()
