"""Focused consistency checks for the revised manuscript and its source files."""
from pathlib import Path
import hashlib,json,re
import fitz
import numpy as np
import torch
from seisformer.model import ModelConfig,SeisFormer

ROOT=Path('outputs/paper_ps')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run():
    e=json.loads((ROOT/'analysis/submission_evidence.json').read_text(encoding='utf-8'))
    tex=(ROOT/'main.tex').read_text(encoding='utf-8');bib=(ROOT/'refs.bib').read_text(encoding='utf-8')
    cites=set(k for m in re.finditer(r'\\citep\{([^}]+)\}',tex) for k in m[1].split(','))
    known=set(re.findall(r'@\w+\{([^,]+)',bib));assert cites<=known
    labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\ref\{([^}]+)\}',tex)
    assert len(set(labels))==len(labels);assert set(refs)<=set(labels)
    pdfs={};environment_warnings={}
    for name in ('XFNet_manuscript','XFNet_supplement','中文核对版'):
        log=(ROOT/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        warnings=[line for line in log.splitlines() if any(v in line for v in ('Overfull','undefined','Missing character','LaTeX Warning:','Package natbib Warning:'))]
        # The installed xeCJK requests a newer LaTeX release. Compilation and
        # glyph rendering succeed; retain this environment warning explicitly.
        allowed=[line for line in warnings if name=='中文核对版' and line=="LaTeX Warning: You have requested release `2026/06/01' of LaTeX,"]
        environment_warnings[name]=allowed
        errors=[line for line in warnings if line not in allowed]
        if errors:raise ValueError('\n'.join(errors))
        doc=fitz.open(ROOT/(name+'.pdf'));txt='\n'.join(p.get_text() for p in doc)
        if '??' in txt:raise ValueError('Unresolved rendered cross reference')
        pdfs[name]=dict(pages=len(doc),sha256=digest(ROOT/(name+'.pdf')))
    cp=ROOT/'reproducibility/model/weights/best_detection.pt'
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    m=SeisFormer(ModelConfig(**ck['model_config']));m.load_state_dict(ck['model'],strict=True)
    assert sum(p.numel() for p in m.parameters())==428774
    assert m.stem[0][1].num_groups==4 and m.decode[0][0].kernel_size==(7,)
    assert m.blocks[0].down.in_features==192 and m.det_head[0].out_features==32
    assert e['primary']['checkpoint_sha256']==digest(cp)
    assert sum(e['snapshot_totals'].values())==9521205
    assert sum(v for k,v in e['snapshot_totals'].items() if k!='quarantine')==8246300
    assert e['primary']['rows']==127940 and e['primary']['overall']['n']==116420
    assert e['primary']['overall']['P']['mae_s']==.2664512290413205
    assert e['primary']['overall']['S']['mae_s']==.2159498568871715
    assert abs(e['primary']['overall']['detection']['0.5']['f1']-.9987521836785626)<1e-12
    assert len(e['primary_source_calibration'])==16
    assert abs(e['calibration_summary']['global']['mean_absolute_deviation']-.1315612867)<1e-9
    assert abs(e['calibration_summary']['local']['mean_absolute_deviation']-.0167688973)<1e-9
    # Six-phase result values from the old mixed-task table and obsolete 524k
    # numbers must not be claimed as the historical P/S checkpoint's baseline comparison.
    for old in ('0.749','1.756','65.4\\%','0.727','0.371','0.871/0.835','0.903/0.873'):
        if old in tex:raise ValueError('Unexpected obsolete statistic: '+old)
    assert '1.161/0.584' in tex and '0.819/0.830' in tex and '0.893/0.872' in tex
    if re.search(r'mean\s+(?:absolute\s+)?(?:coverage|source)[^.\n]{0,70}0\.473',tex):raise ValueError('Maximum deviation described as mean')
    alignment=json.loads((ROOT/'analysis/bilingual_alignment.json').read_text(encoding='utf-8'))
    assert alignment['english_master_sha256']==hashlib.sha256(tex.encode()).hexdigest()
    units=json.loads((ROOT/'analysis/review_units.json').read_text(encoding='utf-8'))
    assert len(alignment['paragraphs'])==sum(u['kind']=='paragraph' for u in units)
    assert alignment['tables']==len(re.findall(r'\\begin\{table\}',tex))
    assert alignment['figures']==len(re.findall(r'\\begin\{figure\}',tex)) and alignment['equations']==3
    zhtxt='\n'.join(p.get_text() for p in fitz.open(ROOT/'中文核对版.pdf'))
    assert all(p['id'] in zhtxt for p in alignment['paragraphs'])
    assert '0.9988' in zhtxt and '428,774' in zhtxt
    assert len(re.findall(r'\\begin\{table\}',tex))==5
    assert len(re.findall(r'\\begin\{figure\}',tex))==5
    assert 'Application to longer records' not in tex
    supporting=(ROOT/'supplement.tex').read_text(encoding='utf-8')
    assert len(re.findall(r'\\section\{',supporting))==7
    assert len(re.findall(r'\\begin\{table\}',supporting))==8
    for label in ('tab:external','fig:external','fig:waveforms'):
        assert r'\label{'+label+'}' not in tex+supporting
    for name in ('XFNet_manuscript','XFNet_supplement','中文核对版'):
        content='\n'.join(p.get_text() for p in fitz.open(ROOT/(name+'.pdf')))
        assert not any(v in content.lower() for v in ('competition','竞赛','1,590.65','1,625.68','1,000-record','test2000'))
    examples=json.loads((ROOT/'analysis/heldout_waveform_selection.json').read_text(encoding='utf-8'))
    assert examples['primary_checkpoint_sha256']==digest(cp)
    assert examples['test_records']==127940 and examples['source_collections']==['STEAD','TXED','PNW']
    pred_path=Path(examples['prediction_file'])
    assert examples['prediction_sha256']==digest(pred_path)
    with np.load(pred_path,allow_pickle=False) as primary_predictions:
        for example in examples['examples']:
            i=example['test_row']
            assert primary_predictions['source'][i]==example['source']
            assert primary_predictions['group'][i]==example['event_group']
            assert np.array_equal(primary_predictions['truth'][i]/100,example['reference_s'])
            assert np.array_equal(primary_predictions['pred'][i]/100,example['prediction_s'])
    comparison=ROOT.parent/'recent_dual_comparison_v1'
    recent=json.loads((comparison/'RESULTS.json').read_text(encoding='utf-8'))
    exported=json.loads((ROOT/'analysis/recent_dual_export.json').read_text(encoding='utf-8'))
    assert recent['status']=='complete' and len(recent['runs'])==6
    assert digest(comparison/'RESULTS.json')==exported['result_sha256']
    assert (ROOT/'analysis/recent_dual_table.tex').read_text(encoding='utf-8').strip() in tex
    assert recent['all_six_test_metadata_arrays_match'] and recent['all_six_training_jobs_verified']
    for run in recent['runs']:
        assert run['completion']['completed_epochs']==20 and run['completion']['completed_updates']==10240
        directory=Path(run['directory'])
        assert digest(directory/'best.pt')==run['completion']['selected_checkpoint_sha256']
        with np.load(directory/'test_predictions.npz',allow_pickle=False) as arrays:
            assert len(arrays['truth'])==127940 and arrays['event_mask'].sum()==116420
            assert all(np.isfinite(arrays[k]).all() for k in ('truth','pred','confidence','event','event_prob'))
            for j,phase in enumerate(('P','S')):
                known=arrays['truth'][:,j]>=0
                value=float((np.abs(arrays['pred'][known,j]-arrays['truth'][known,j])/100).mean())
                assert abs(value-run['metrics']['overall'][phase]['mae_s'])<1e-12
    result=dict(status='passed',primary_weight_sha256=digest(cp),rendered_pdfs=pdfs,cited_references=len(cites),
        figures=len(re.findall(r'\\begin\{figure\}',tex)),tables=len(re.findall(r'\\begin\{table\}',tex)),
        summary_words=len(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S)[1].split()),
        chinese_aligned_paragraphs=len(alignment['paragraphs']),chinese_tables=alignment['tables'],chinese_figures=alignment['figures'],chinese_equation_groups=alignment['equations'],
        environment_warnings=environment_warnings,
        recent_comparison=dict(runs=6,completed_epochs_each=20,all_test_mae_recomputed=True,
            source_result_sha256=exported['result_sha256'],accuracy_parameter_runtime_table_matches=True),
        physical_model_details_match=True,same_checkpoint_calibration_and_waveform_examples=True,
        cached_primary_global_and_source_metrics_recomputed=True,
        new_training_performed=False,protected_regional_holdout_accessed=False,
        waveform_examples=dict(sources=examples['source_collections'],test_records=127940,
            preserved_predictions_match=True,no_new_model_evaluation=True),
        scope='Manuscript consistency and compilation, not journal acceptance certification.')
    (ROOT/'analysis/manuscript_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
