"""Integrate the completed matched comparison into the English manuscript."""
from pathlib import Path

from scripts.recent_dual_core import PROJECT,read

PAPER=PROJECT/'outputs/paper_ps'


def run():
    result=read(PROJECT/'outputs/recent_dual_comparison_v1/RESULTS.json')
    assert result['status']=='complete' and len(result['runs'])==6
    path=PAPER/'main.tex';tex=path.read_text(encoding='utf-8')
    if r'\label{tab:recent}' in tex:raise ValueError('Recent comparison is already integrated.')
    def replace(old,new):
        nonlocal tex
        assert tex.count(old)==1,old[:100]
        tex=tex.replace(old,new)
    replace('0.473/0.313~s. With 1,048,576 training records,',
        '0.473/0.313~s. A separate matched-budget comparison from random initialization gives P/S MAE 0.425/0.295~s, versus 0.751/0.376~s for TwoStream-EQT, with 31.1~per~cent fewer parameters. With 1,048,576 training records,')
    replace("The paired temporal control tests that implementation's incremental timing benefit; it does not infer superiority from parameter count alone. Comparisons with the closest dual-domain methods would require their own native preprocessing and matched evaluation.",
        'The paired temporal control isolates the incremental timing effect of the spectral paths. An additional matched-budget comparison with the public cross-attention configuration of TwoStream-EQT places the complete implementation against a recent dual-domain reference, retaining native input processing and measuring arrival error, parameter count and runtime together.')
    methods=r'''For the recent-model comparison, the public TwoStream-EQT network retains seven convolutional stages, seven residual blocks and three LSTM blocks in its time and frequency paths \citep{peng2025twostream}. An adapter converts ENZ to ZNE and applies its native component demeaning, shared standard-deviation normalization and six-sample taper. Both models use the same Gaussian P/S targets, weighting and unknown/invalid-time masks. The reference retains dense event supervision between P and P+1.4(S-P), masking incomplete known intervals, while SeisFormer retains its window-level detector. Reference LSTM operations use FP32 within mixed-precision training; all final evaluation uses FP32 with TF32 disabled.

Both comparison architectures start from random weights and use the same 262,144 training records and 16,384 validation records. Each receives a search over learning rates $10^{-4}$, $3\times10^{-4}$ and $10^{-3}$ on a shared 32,768-record training subset and 4,096 validation records, with three epochs per candidate. Validation mean P/S MAE selects $10^{-3}$ for both models. Fresh runs then use seeds 42/43/44, 20 epochs, batch 512, AdamW, weight decay 0.01, clipping at norm 1 and cosine decay to one tenth of the initial rate. Search weights are not inherited. The checkpoint rule is minimum full-validation mean P/S MAE. Complete protocol and seed results are given in Supplementary Section~S7.

'''
    replace(r'\subsection{Picking, uncertainty and longer-record inference}',methods+r'\subsection{Picking, uncertainty and longer-record inference}')
    replace('These event and arrival metrics describe different tasks; the peak scores are not used as calibrated timing probabilities.',
        'These event and arrival metrics describe different tasks; the peak scores are not used as calibrated timing probabilities. For the recent-model comparison, we additionally select an event threshold allowing at most 1~per~cent of known validation noise and choose phase thresholds by validation F1. These operating points are frozen before test evaluation.')
    comparison=r'''\subsection{Matched comparison with a recent dual-domain picker}
With fresh initialization and matched training budgets, SeisFormer gives P/S MAE $0.4250\pm0.0060$/$0.2951\pm0.0071$~s, compared with $0.7511\pm0.0424$/$0.3764\pm0.0137$~s for TwoStream-EQT (Table~\ref{tab:recent}). Mean reductions are 43.4 and 21.6~per~cent, with the same direction for both phases in all three paired seeds. A joint paired bootstrap over model seeds and event groups gives 95~per~cent intervals of $[-0.3906,-0.2664]$~s for the P-error difference and $[-0.1005,-0.0645]$~s for S, where differences are SeisFormer minus the reference. Both intervals are below zero under this common data and training budget.

'''+(PAPER/'analysis/recent_dual_table.tex').read_text(encoding='utf-8')+r'''
SeisFormer uses 428,774 parameters against 622,668 for the reference, a reduction of 31.1~per~cent. Single-window GPU forward time is 3.66 versus 9.91~ms, and CPU time is 12.14 versus 14.65~ms with two threads. At batch 64, the reference has higher throughput: 5,436 versus 4,880 windows~s$^{-1}$. Thus single-window latency and batch throughput describe different computational trade-offs. Under the validation-derived operating points, P/S picking F1 is 0.8773/0.8641 for SeisFormer and 0.8514/0.8334 for TwoStream-EQT. All seed results, fixed-threshold metrics and learning-rate screens are retained in the supporting data.

'''
    replace(r'\subsection{Source-dependent timing and physical interpretation}',comparison+r'\subsection{Source-dependent timing and physical interpretation}')
    replace('0.3 phase-probability threshold','0.3 phase-score threshold')
    tex=tex.replace('peak probability','peak score')
    replace('Hardware, software and precision settings are recorded in Supplementary Table~S6.',
        r'Hardware, software and precision settings are recorded in Supplementary Table~S6. The separate matched-reference timings in Table~\ref{tab:recent} use alternating model order and explicitly disable TF32.')
    replace("In the context of existing dual-domain pickers \\citep{peng2025twostream,tang2026seismodual}, this is evidence for a compact realization and its incremental benefit under a specified recipe. The broader-data continuation improves the same task to P/S MAE 0.266/0.216~s. The representation contrast and the attainable larger-data result answer different questions, and neither requires attributing all gains to a new attention mechanism.",
        'The matched comparison with TwoStream-EQT adds a recent dual-domain reference: arrival errors are lower while the complete model uses 31.1~per~cent fewer parameters. Single-window latency and batch throughput expose different operating trade-offs rather than a uniform speed ranking. The broader-data continuation reaches P/S MAE 0.266/0.216~s. Together, the representation control, recent-method comparison and larger-data sequence connect the compact design to measured arrival estimates.')
    replace('The closest additional representation comparisons are now TwoStream-EQT and SeismoDual, with LFTNet supplying a recent lightweight joint-task reference \\citep{peng2025twostream,tang2026seismodual,guo2025lftnet}. Their published scores cannot be ranked directly against this mixed-source cohort. Matched adaptation under their native input conventions would address relative method performance.',
        'The additional from-scratch experiment supplies a matched comparison with the public cross-attention configuration of TwoStream-EQT. Other dual-domain and compact designs, including SeismoDual and LFTNet \\citep{tang2026seismodual,guo2025lftnet}, offer extensions to this evaluation under their native input conventions.')
    replace('The 5,856-parameter spectral extension reduces mean P/S errors by 0.078/0.033~s across three paired fine-tuning seeds.',
        'The 5,856-parameter spectral extension reduces mean P/S errors by 0.078/0.033~s across three paired fine-tuning seeds. In the separate from-scratch comparison, the model reduces P/S MAE by 43.4/21.6~per~cent relative to TwoStream-EQT under the matched budget, with 31.1~per~cent fewer parameters and lower single-window latency.')
    replace('These findings support this compact realization of time--frequency picking; the remaining tests are a matched comparison with recent dual-domain models and a documented catalogue or location-level application.',
        'Together, these findings support a compact single-station estimator with controlled timing gains and measured accuracy--cost trade-offs for subsequent catalogue-processing workflows.')
    replace('Supplementary material describes the training histories, exploratory architecture comparisons, polarization continuations, source calibration, timing measurements and the provenance of each result.',
        'Supplementary material describes the training histories, exploratory architecture comparisons, polarization continuations, source calibration, timing measurements, the matched recent-model protocol and all seed and learning-rate results.')
    path.write_text(tex,encoding='utf-8')
    print('Integrated completed six-run comparison into main.tex.')


if __name__=='__main__':run()
