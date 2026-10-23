"""Refresh the local v7 manuscript/reproduction package after successful checks."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'outputs/paper_ps'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def run():
    check=json.loads((ROOT/'analysis/manuscript_checks.json').read_text(encoding='utf-8'))
    assert check['status']=='passed'
    for name,record in check['rendered_pdfs'].items():
        assert sha(ROOT/(name+'.pdf'))==record['sha256'], 'Re-run manuscript checks after changing PDFs.'
    alignment=json.loads((ROOT/'analysis/bilingual_alignment.json').read_text(encoding='utf-8'))
    # Alignment hashes the decoded text with normalized newlines; package
    # manifest entries below hash raw bytes, including Windows CRLF if present.
    assert hashlib.sha256((ROOT/'main.tex').read_text(encoding='utf-8').encode()).hexdigest()==alignment['english_master_sha256']
    snapshots=ROOT/'reproducibility/source_scripts'
    snapshots.mkdir(exist_ok=True)
    for name in ('benchmark_inference','build_paper_ps_tex','build_ps_supplement',
                 'check_ps_submission','compile_ps_submission','export_ps_manuscript_markdown',
                 'make_ps_submission_figures','make_ps_heldout_waveforms','prepare_ps_submission_evidence',
                 'build_ps_chinese_review','search_ps_journal_literature','package_ps_submission',
                 'recent_dual_core','prepare_recent_dual_comparison','preflight_recent_dual_comparison',
                 'train_recent_dual_comparison','run_recent_dual_comparison','analyze_recent_dual_comparison',
                 'export_recent_dual_paper_table','export_recent_dual_repro','integrate_recent_dual_manuscript',
                 'rebase_ps_review_translation','diagnose_recent_dual_nan','probe_recent_dual_precision'):
        shutil.copy2(PROJECT/'scripts'/(name+'.py'),snapshots/(name+'.py'))
    files=[]
    for name in ('main.tex','supplement.tex','chinese_review.tex','refs.bib',
                 'SeisFormer_manuscript_v7.pdf','SeisFormer_supplement_v7.pdf','中文核对版_v7.pdf',
                 'MANUSCRIPT_EN.md','HIGHLIGHTS.md','中文核对版.md','中英逐段对照.md','README.md',
                 'COVER_LETTER_GJI.md','COVER_LETTER_CG.md','修改说明_GJI.md',
                 '投稿信息待填写.md','选刊与论文定位.md',
                 '论文模式修改对照_20261022.md','近期同类论文对照与投稿判断_20261022.md',
                 '近期双域公平对照结果.md',
                 'analysis/manuscript_checks.json','analysis/primary_runtime.json',
                 'analysis/references/crossref_metadata.json','analysis/submission_evidence.json',
                 'analysis/heldout_waveform_selection.json','analysis/heldout_waveform_data.npz',
                 'analysis/review_units.json','analysis/review_translation_zh.json',
                 'analysis/bilingual_alignment.json','analysis/build_logs/compilation.json',
                 'analysis/review_labels_zh.json','analysis/review_v7_overrides_zh.json',
                 '测试资料与波形示例修订_v7.md',
                 'analysis/recent_dual_table.tex','analysis/recent_dual_supplement.tex','analysis/recent_dual_export.json'):
        files.append(ROOT/name)
    main=(ROOT/'main.tex').read_text(encoding='utf-8')
    figure_files=[]
    for figure in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',main):
        figure_files.extend((ROOT/figure).with_suffix(ext) for ext in ('.pdf','.svg','.png'))
    files.extend(figure_files)
    files.extend(p for p in (ROOT/'reproducibility').rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.name!='revise_ps_competition_scope.py')
    files.extend(p for p in (ROOT/'analysis/journal_recent_20261022').iterdir()
                 if p.suffix in ('.json','.md','.txt'))
    files=sorted(set(files),key=lambda p:p.relative_to(ROOT).as_posix())
    records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    manifest=dict(version_date='2026-10-23',manuscript_revision='v7-cg-ready',
                  target_journal='Computers & Geosciences',
                  alternative_journal='Geophysical Journal International',
                  purpose='Local manuscript/reproduction materials; no external submission or publication performed',
                  files=records,primary_model_sha256=check['primary_weight_sha256'])
    manifest_path=ROOT/'SeisFormer_source_manifest_v7.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    target=ROOT/'SeisFormer_submission_materials_v7_20261022.zip'
    temporary=target.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in files+[manifest_path]: archive.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for record in records:
            assert hashlib.sha256(archive.read(record['path'])).hexdigest()==record['sha256']
    temporary.replace(target)
    cg_files=sorted(set([
        ROOT/'main.tex', ROOT/'supplement.tex', ROOT/'refs.bib',
        ROOT/'SeisFormer_manuscript_v7.pdf', ROOT/'SeisFormer_supplement_v7.pdf',
        ROOT/'HIGHLIGHTS.md', ROOT/'COVER_LETTER_CG.md',
    ] + figure_files), key=lambda p:p.relative_to(ROOT).as_posix())
    cg_target=ROOT/'SeisFormer_CG_submission_upload_v7.zip'
    cg_tmp=cg_target.with_suffix('.tmp.zip')
    with zipfile.ZipFile(cg_tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in cg_files: archive.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(cg_tmp) as archive:
        assert archive.testzip() is None
    cg_tmp.replace(cg_target)
    print(json.dumps(dict(package=str(target),cg_upload_package=str(cg_target),files=len(records)+1,cg_upload_files=len(cg_files),MiB=round(target.stat().st_size/2**20,2),
                          all_archive_hashes_verified=True),ensure_ascii=False,indent=2))


if __name__=='__main__':run()
