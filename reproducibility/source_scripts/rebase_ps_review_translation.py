"""Carry unchanged v4 paragraph translations into v5 and require explicit edits."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.build_ps_chinese_review import ROOT,units_from_markdown,protect_display_math


def normalized(text):
    return re.sub(r'\b(Table|Figure|Fig\.)\s+(\d+)',r'\1 NUMBER',text)


def run(overrides_path, archive=None):
    archive=Path(archive) if archive else ROOT/'archive_before_recent_dual_comparison_20261022'
    old_units=json.loads((archive/'analysis/review_units.json').read_text(encoding='utf-8'))
    old_zh=json.loads((archive/'analysis/review_translation_zh.json').read_text(encoding='utf-8'))
    lookup={normalized(u['english']):u for u in old_units if u['kind']=='paragraph'}
    units=units_from_markdown(protect_display_math((ROOT/'MANUSCRIPT_EN.md').read_text(encoding='utf-8')))
    overrides=json.loads(Path(overrides_path).read_text(encoding='utf-8')) if Path(overrides_path).exists() else {}
    translated={};missing=[]
    for unit in units:
        if unit['kind']!='paragraph':continue
        pid=unit['id'];english=unit['english'];digest=hashlib.sha256(english.encode()).hexdigest()
        override=overrides.get(pid)
        if override:
            assert override['english_sha256']==digest, 'Override belongs to a different English paragraph: '+pid
            translated[pid]=override['chinese'];continue
        old=lookup.get(normalized(english))
        if old:
            text=old_zh[old['id']]
            prior=re.findall(r'\bTable\s+(\d+)',old['english'])
            current=re.findall(r'\bTable\s+(\d+)',english)
            assert len(prior)==len(current)
            mapping=dict(zip(prior,current))
            text=re.sub(r'表(\d+)',lambda m:'表'+mapping.get(m[1],m[1]),text)
            translated[pid]=text
        else:missing.append(dict(id=pid,section=unit['section'],english=english,english_sha256=digest))
    if missing:
        (ROOT/'analysis/review_translation_missing.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(missing,ensure_ascii=False,indent=2))
        raise ValueError(f'{len(missing)} new or revised paragraphs require explicit Chinese translations.')
    (ROOT/'analysis/review_translation_zh.json').write_text(json.dumps(translated,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'analysis/review_units.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Rebased {len(translated)} paragraphs; {len(overrides)} explicit translation overrides.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overrides',default=str(ROOT/'analysis/review_v5_overrides_zh.json'))
    p.add_argument('--archive')
    args=p.parse_args()
    run(args.overrides,args.archive)
