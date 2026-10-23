"""Build supplementary tables directly from the collected P/S evidence."""
from pathlib import Path
import json
import numpy as np

ROOT=Path('outputs/paper_ps')


def run():
    e=json.loads((ROOT/'analysis/submission_evidence.json').read_text(encoding='utf-8'))
    s=[r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,graphicx,booktabs,tabularx,amsmath}
\usepackage{hyperref}
\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\title{Supplementary data: SeisFormer compact time--frequency phase picking}
\author{}\date{}
\begin{document}\maketitle
\section{Training histories and checkpoint identity}
All three controlled 262k time-domain/dual-domain pairs use the same time-domain pilot initialization. A seed denotes stochastic fine-tuning from this shared parent, not a fully independent pilot. The primary 1M seed-42 checkpoint has SHA256
\begin{quote}\small\texttt{08afa0d899bbc03e5680dff2b49768aacb1ce5fdfe07525d037b1c87e9ff3342}.\end{quote}
Its stored epoch index is16 (human epoch17), with278,528 optimizer updates in the1M stage. The stage starts from the524k model selected by validation phase MAE; the524k stage starts from the32k pilot. Stage-wise example counts exclude these ancestors.

Additional1M seed runs share the same524k parent. Seed43 was manually stopped during epoch13; twelve complete epoch reports are retained. Seeds42 and44 completed20. Their available test results are preserved in Table~\ref{tab:seeds}; cross-run summaries do not imply equal training budgets. The primary result is the archived seed42 detection-selected state, not the minimum test MAE found among these rows.
\begin{table}[htbp]\centering\small
\caption{Available1M checkpoint results. M: validation phase-MAE selection; D: validation detection-F1 selection. Completed epochs describe the full run, not the selected epoch.}
\label{tab:seeds}
\begin{tabular}{lrrrrrr}\toprule
Seed & Completed epochs & Rule & Selected epoch & P MAE(s) & S MAE(s) & Det F1\\\midrule''']
    for seed,name,completed in ((42,'standardized_dual_1m_20ep',20),(43,'standardized_dual_1m_seed43',12),(44,'standardized_dual_1m_seed44',20)):
        for rule,key in (('M','eval_test'),('D','eval_test_det')):
            r=e['runs'][name][key];m=r['overall']
            s.append(f"{seed} & {completed} & {rule} & {r['checkpoint_epoch']+1} & {m['P']['mae_s']:.4f} & {m['S']['mae_s']:.4f} & {m['detection']['0.5']['f1']:.6f} \\")
    s.append(r'''\bottomrule\end{tabular}\end{table}
\section{Exploratory architecture comparisons}
The historical shared-tensor comparison trains models on standardized ENZ/shared-RMS inputs and the524k row list, evaluating the same127,940-record test list. The from-scratch U-Net retains twelve complete epoch reports and was stopped during epoch13 of20 planned epochs; its selected checkpoint is epoch12. The official PhaseNet fine-tuning run and SeisFormer each completed20. These budgets and input conventions make the experiments useful shared-recipe checks, but they do not replace a comparison respecting each pretrained model's native normalization and component order. They are reported as exploratory results and do not support a universal method ranking.
\begin{table}[htbp]\centering\small
\caption{Shared-input exploratory comparisons. Parameters for PhaseNet refer to the official backbone; model-specific fine-tuning and preprocessing need to be considered when interpreting these rows.}
\begin{tabular}{lrrrrrr}\toprule
Model & Params(M) & Epochs & P MAE(s) & S MAE(s) & P F1 & S F1\\\midrule''')
    for label,name,param,ep in (('SeisFormer dual','standardized_dual_expanded_524k',.429,20),('Scratch U-Net','standardized_unet_524k',1.56,12),('Official PhaseNet, shared-input FT','standardized_phasenet_524k',.27,20)):
        r=e['runs'][name]['eval_test']['overall']
        s.append(f"{label} & {param:.3f} & {ep} & {r['P']['mae_s']:.3f} & {r['S']['mae_s']:.3f} & {r['P']['picking_at_0.3']['f1']:.3f} & {r['S']['picking_at_0.3']['f1']:.3f} \\")
    s.append(r'''\bottomrule\end{tabular}\end{table}
The cleanest architecture contrast is the paired temporal control in main Table2. The two spectral paths are added together; separate Morlet-only and STFT-only experiments have not been established by that contrast. Similarly, a parameter count is not a FLOP count: the FFT/filterbank computations add runtime despite the small number of learned weights.

\section{Polarization-biased continuations}
For the polarization variant,101-sample centred waveform windows at16-sample spacing give a3-by-3 covariance matrix. Rectilinearity and the six independent entries of the principal-eigenvector outer product form seven sign-invariant features. An MLP maps them to96 dimensions; normalized feature similarities form an additive attention bias with a learnable initial scale0.1. The implementation disables the polarization embedding/bias for an entire record if any component is identically zero. This is a record-level guard, not a per-token missingness estimator.

The524k variant starts from the fitted524k dual-domain model and completes20continuation epochs, selecting epoch11 by phase MAE. The1M variant starts from the primary1M checkpoint with a planned10-epoch continuation; five complete epochs are recorded before stopping during epoch6, and detection selection retains epoch5. Table~\ref{tab:ppaa} preserves the recorded paired event-group bootstrap comparisons. Since an otherwise identical additional-training control was not run, the difference includes both the added module and further optimization. It cannot isolate the causal effect of polarization or prove its substitution by learned physics.
\begin{table}[htbp]\centering\small
\caption{Polarization continuation comparisons. Confidence intervals resample paired event groups; the reference and continuation states have different accumulated update histories.}
\label{tab:ppaa}
\begin{tabular}{lrrrr}\toprule
Stage & Reference P MAE(s) & Continued P MAE(s) & Difference(s) &95\%interval(s)\\\midrule
524k &0.3871&0.3472&$-0.0400$&$[-0.0524,-0.0267]$\\
1M &0.2665&0.2629&$-0.0036$&$[-0.0133,0.0064]$\\
\bottomrule\end{tabular}\end{table}

\section{Source calibration and local-sample sensitivity}
All entries in Table~\ref{tab:calibration} use the same primary checkpoint, its127,623 calibration-list records and its127,940 test-list records. The empirical quantile for each source/phase uses only known calibration targets. Source intervals retain their measured radii even when the target distribution is concentrated at a fixed offset. A zero radius for NEIC indicates that calibration residuals are heavily concentrated atzero; it should not be interpreted as zero physical timing uncertainty.
\begin{table}[htbp]\centering\small
\caption{Pooled and source-specific calibration at nominal90\%coverage. Target and group counts describe test support; calibration counts are supplied in the accompanying JSON.}
\label{tab:calibration}
\begin{tabular}{llrrrrr}\toprule
Source & Phase & Targets & Groups & Pooled coverage & Local radius(s) & Local coverage\\\midrule''')
    for r in sorted(e['primary_source_calibration'],key=lambda r:(r['source'],r['phase'])):
        s.append(f"{r['source']} & {r['phase']} & {r['n_test']:,} & {r['test_groups']:,} & {r['global_coverage']:.3f} & {r['local_radius']:.3f} & {r['local_coverage']:.3f} \\")
    s.append(r'''\bottomrule\end{tabular}\end{table}
Mean absolute coverage deviation is the arithmetic mean of$|\widehat C_{sj}-0.9|$ across the16source--phase groups; maximum deviation is their maximum. The pooled/local means are0.13156/0.01677; maxima are0.47283/0.08254. These definitions prevent the mean and worst-group summaries from being interchanged.

As a sensitivity calculation, for each target source we sample$n$local calibration residuals without replacement and combine them with200residuals sampled from the other sources, also without replacement. We repeat30times using seed42. Budgets greater than the available local residual count are skipped. Table~\ref{tab:budget} gives mean P coverage for three harder sources; all phases/sources and10th--90th percentile ranges are in the accompanying data. These are record-subsampling curves on already represented sources, not a validated new-network sample-size rule. In particular,250--1,000local labels are not a universal coverage guarantee.
\begin{table}[htbp]\centering\small
\caption{P coverage under the local-calibration sensitivity calculation; nominal coverage is0.9.}
\label{tab:budget}\begin{tabular}{lrrrrrr}\toprule
Source &0local&25&100&250&500&1,000\\\midrule''')
    for source in ('CREW','GEOFON','TXED'):
        rr={r['n_local']:r['mean_coverage'] for r in e['calibration_budget'] if r['source']==source and r['phase']=='P'}
        s.append(source+' & '+' & '.join(f'{rr[n]:.3f}' for n in (0,25,100,250,500,1000))+r' \\')
    s.append(r'''\bottomrule\end{tabular}\end{table}

\section{Runtime}
The primary checkpoint is loaded strictly before measurement. The benchmark uses PyTorch2.14.0+cu130, an NVIDIA RTX5070Ti and two CPU threads (AMD64Family26Model68). Inputs are resident on the device; ten warm-up passes precede60timed forwards, with synchronization around the GPU timing interval. Table~\ref{tab:runtime} reports total batch time and throughput separately. BF16 is supplementary timing only; its accuracy was not used to replace the primary checkpoint's reported results. The older timing artifact used a524k checkpoint and a different run environment; it is not substituted for this direct primary-model measurement.
\begin{table}[htbp]\centering\small
\caption{Primary-model forward-only timing,6,000sample inputs. CPU uses two threads.}
\label{tab:runtime}\begin{tabular}{llrrr}\toprule
Device & Precision & Batch & Batch time(ms) & Windows/s\\\midrule''')
    runtime=json.loads((ROOT/'analysis/primary_runtime.json').read_text(encoding='utf-8'))
    for r in runtime['benchmarks']:
        s.append(f"{r['device'].upper()} & {r['autocast'].upper()} & {r['batch']} & {r['seconds_per_batch']*1000:.3f} & {r['windows_per_second']:.1f} \\")
    s.append(r'''\bottomrule\end{tabular}\end{table}
\section{Reproducibility}
The reproduction bundle contains model and data-list identities, cached predictions and the scripts that generate the revised tables/figures. Key files are \texttt{submission\_evidence.json}, \texttt{primary\_runtime.json} and \texttt{heldout\_waveform\_selection.json}. Main figures use the preserved P/S model only. Results from later six-phase or long-record prototypes are separate experiments and are not attributed to this checkpoint.
\end{document}''')
    # Fill ordinary word--number spacing from the compact source strings above.
    import re
    text='\n'.join(s)+'\n'
    cleaned=[]
    for line in text.splitlines():
        if not line.startswith('\\') and '&' not in line and len(line)>80:
            line=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', line)
            line=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', line)
            for a,b in [('F 1','F1'),('FP 32','FP32'),('BF 16','BF16'),('SHA 256','SHA256'),('cu 130','cu130')]:line=line.replace(a,b)
        # Generated numeric rows need an actual TeX row terminator.
        if line.endswith('\\') and not line.endswith('\\\\'):line+='\\'
        cleaned.append(line)
    text='\n'.join(cleaned)+'\n'
    for a,b in [('1 M','1M'),('262 k','262k'),('524 k','524k'),('32 k','32k'),('1M','1M'),
                ('Available1M','Available 1M'),('25 mm','25mm'),('MAE(s)','MAE (s)'),
                ('Params(M)','Params (M)'),('time(ms)','time (ms)'),('95\\%interval','95\\% interval'),
                ('nominal90\\%coverage','nominal 90\\% coverage'),('same127,','same 127,'),
                ('same524k','same 524k'),('same32k','same 32k'),('two threads.','two threads.'),
                ('timing,6,000sample','timing, 6,000-sample'),('errors$','errors $'),('$are','$ are'),
                ('With$','With $'),('$records','$ records'),('and$','and $'),('$emitted','$ emitted'),
                ('Official PhaseNet, shared-input FT','PhaseNet, shared-input FT')]:text=text.replace(a,b)
    recent=ROOT/'analysis/recent_dual_supplement.tex'
    if recent.exists():
        text=text.replace('\\begin{document}',r'\makeatletter\setlength{\@fptop}{0pt}\setlength{\@fpsep}{18pt}\makeatother'+'\n'+r'\begin{document}')
        recent_text=recent.read_text(encoding='utf-8')
        recent_text=recent_text.replace(r'\section*{S7\quad Matched recent dual-domain comparison}',r'\section{Matched recent dual-domain comparison}')
        text=text.replace('\\end{document}',recent_text+'\n\\end{document}')
    (ROOT/'supplement.tex').write_text(text,encoding='utf-8')
    print(ROOT/'supplement.tex')


if __name__=='__main__':run()
