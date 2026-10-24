"""Export TeX tables only after all matched trainings and evaluations complete."""
from pathlib import Path
import json

from scripts.recent_dual_core import PROJECT,read,sha

EXPERIMENT=PROJECT/'outputs/recent_dual_comparison_v1'
PAPER=PROJECT/'outputs/paper_ps'


def run():
    result=read(EXPERIMENT/'RESULTS.json')
    assert result['status']=='complete' and result['all_six_training_jobs_verified']
    assert result['all_six_test_metadata_arrays_match'] and len(result['runs'])==6
    summary=result['summary'];timing=result['runtime']['results']
    def runtime(architecture,device,batch,key):
        return next(r[key] for r in timing if r['architecture']==architecture and r['device']==device and r['batch']==batch)
    rows=[]
    for architecture,name in [('seisformer','XFNet'),('twostream_eqt','TwoStream-EQT')]:
        r=summary[architecture]
        rows.append(f"{name} & ${r['P']['mae_s_mean']:.4f}\\pm{r['P']['mae_s_sd']:.4f}$ & ${r['S']['mae_s_mean']:.4f}\\pm{r['S']['mae_s_sd']:.4f}$ & {r['parameters']:,} & {runtime(architecture,'cuda',1,'milliseconds_per_batch'):.2f} & {runtime(architecture,'cuda',64,'windows_per_second'):,.0f} & {runtime(architecture,'cpu',1,'milliseconds_per_batch'):.2f}\\\\")
    table=r'''\begin{table}[tb]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Matched-budget comparison with the public cross-attention configuration of TwoStream-EQT. Both models are trained from scratch on the same 262k records for 20 epochs using three seeds. Arrival errors are mean $\pm$ sample SD across seeds. Runtime uses the selected seed-42 models, FP32 with TF32 disabled, alternating model execution order, the same RTX 5070 Ti and two CPU threads (accounting for the slight difference from the standalone 4.18~ms primary-checkpoint timing in Section~\ref{sec:runtime}); B1/B64 denote batch sizes. Native spectral computation and input adaptation are included.}
\label{tab:recent}
\begin{tabular}{lrrrrrr}
\toprule
Model & P MAE (s) & S MAE (s) & Parameters & GPU B1 (ms) & GPU B64 (s$^{-1}$) & CPU B1 (ms)\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (PAPER/'analysis/recent_dual_table.tex').write_text(table,encoding='utf-8')
    seed_rows=[]
    for record in sorted(result['runs'],key=lambda r:(r['architecture'],r['seed'])):
        metrics=record['metrics'];completion=record['completion'];cal=metrics['at_validation_operating_point']
        name='XFNet' if record['architecture']=='seisformer' else 'TwoStream-EQT'
        seed_rows.append(f"{name} & {record['seed']} & {completion['selected_epoch']} & {metrics['overall']['P']['mae_s']:.4f} & {metrics['overall']['S']['mae_s']:.4f} & {cal['P']['f1']:.4f} & {cal['S']['f1']:.4f}\\\\")
    selection=read(EXPERIMENT/'selected_learning_rates.json');lr_rows=[]
    for architecture,name in [('seisformer','XFNet'),('twostream_eqt','TwoStream-EQT')]:
        for r in selection[architecture]['candidates']:
            lr_rows.append(f"{name} & {r['learning_rate']:g} & {r['validation_fp32_mean_ps_mae_s']:.4f}\\\\")
    supplement=r'''
\section*{S7\quad Matched recent dual-domain comparison}
The reference is the author-linked public CrossTwoStreamEQT configuration associated with Peng et al. (2025), doi:10.1016/j.cageo.2025.105991. It retains seven convolutional stages, seven residual blocks, three LSTM blocks, STFT and cross-attention fusion. Its model source matches upstream Git blob \texttt{85afad0d97bd2216d5b9af1fd2ade88b276ff9c2}. The adapter converts ENZ to ZNE and uses native component demeaning, shared standard-deviation normalization and a six-sample taper. All 622,668 parameters are retained.

Both comparison models start from random weights. They share the exact 262,144 training records, 16,384 validation records and 127,940 test list; per-record crops and batch order are paired by seed and epoch. Training uses batch 512, AdamW, weight decay 0.01, gradient clipping at 1 and cosine decay to one tenth of the initial learning rate. Each model receives the same three-candidate learning-rate search on a 32,768-record training subset and 4,096 validation records, with three epochs per candidate (Table~\ref{tab:recentlr}). Final runs use seeds 42/43/44 and 20 epochs, without inheriting the search weights. Selection minimizes mean P/S validation MAE.

The Gaussian P/S width, peak weighting and validity/unknown masks are shared with the main model. Each architecture retains its native auxiliary detection output: XFNet uses window-level BCE; the reference uses dense BCE between P and P+1.4(S-P), with incomplete observed P/S intervals masked and confirmed noise assigned zero. In mixed-precision training, reference LSTM operations use FP32 for numerical stability on the recorded runtime. This preserves the released network and its full-FP32 inference. All final test predictions and efficiency measurements use FP32 with TF32 disabled.

The operating point in Table~\ref{tab:recentseeds} uses the same 1 per cent validation-noise budget. Phase thresholds are selected on validation only; unconditional arrival MAE retains every known target. A wrong accepted pick counts as both FP and FN. The complete fixed-threshold, source-wise results and predictions accompany the reproduction data. All six completed training histories and selected checkpoints were verified before test evaluation. The original 1M historical model remains a separate training sequence.

\begin{table}[htbp]
\centering\small
\caption{Matched-comparison results for all seeds. Every final run completed 20 epochs; the selected epoch is determined by validation. Picking F1 uses a 0.3-s tolerance and the validation-derived operating point.}
\label{tab:recentseeds}
\begin{tabular}{lrrrrrr}
\toprule
Model & Seed & Selected epoch & P MAE (s) & S MAE (s) & P F1 & S F1\\
\midrule
'''+ '\n'.join(seed_rows)+r'''
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering\small
\caption{Equal-budget learning-rate screening. Values are FP32 validation mean P/S MAE from the selected three-epoch screen checkpoint, not final test performance.}
\label{tab:recentlr}
\begin{tabular}{lrr}
\toprule
Model & Initial learning rate & Validation P/S MAE (s)\\
\midrule
'''+ '\n'.join(lr_rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (PAPER/'analysis/recent_dual_supplement.tex').write_text(supplement,encoding='utf-8')
    report=dict(result_sha256=sha(EXPERIMENT/'RESULTS.json'),source='Frozen six-run comparison',
        main_table_sha256=sha(PAPER/'analysis/recent_dual_table.tex'),supplement_sha256=sha(PAPER/'analysis/recent_dual_supplement.tex'))
    (PAPER/'analysis/recent_dual_export.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
