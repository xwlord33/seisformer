"""Compile the revised review PDFs with the available native MiKTeX tools."""
from pathlib import Path
import subprocess
import shutil
import json

ROOT=Path(__file__).resolve().parents[1]/'outputs/paper_ps'


def run(only=None):
    out=ROOT/'analysis/build_logs';out.mkdir(exist_ok=True)
    previous=out/'compilation.json'
    reports=[r for r in json.loads(previous.read_text(encoding='utf-8')) if r['source']!=only] if only and previous.exists() else []
    for source,job,bibliography,engine in [('main.tex','XFNet_manuscript',True,'pdflatex'),('supplement.tex','XFNet_supplement',False,'pdflatex'),('chinese_review.tex','中文核对版',False,'xelatex')]:
        if only and source!=only:continue
        commands=[[engine,'-interaction=nonstopmode','-halt-on-error','-synctex=1','-jobname='+job,source]]
        if bibliography:commands.append(['bibtex',job])
        commands.extend([commands[0],commands[0]])
        for i,command in enumerate(commands):
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            (out/f'{job}_{i}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            if result.returncode:
                print(result.stdout[-4500:]);print(result.stderr[-700:]);raise RuntimeError(f'{job} build failed, pass{i}')
        # Page/float changes can require one more cross-reference pass.
        for retry in range(2):
            log=(ROOT/(job+'.log')).read_text(encoding='utf-8',errors='replace')
            if 'Label(s) may have changed' not in log:break
            result=subprocess.run(commands[0],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            (out/f'{job}_rerun{retry}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            if result.returncode:raise RuntimeError(f'{job} cross-reference pass failed')
            commands.append(commands[0])
        reports.append(dict(source=source,pdf=job+'.pdf',status='compiled',commands=commands))
    (out/'compilation.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--only',choices=['main.tex','supplement.tex','chinese_review.tex'])
    run(parser.parse_args().only)
