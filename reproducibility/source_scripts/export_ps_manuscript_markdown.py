"""Keep the readable Markdown manuscript synchronized with the TeX master."""
from pathlib import Path
import re

ROOT=Path('outputs/paper_ps')


def braced(s,start):
    if s[start]!='{':raise ValueError('Expected opening brace')
    level=1;i=start+1
    while i<len(s) and level:
        if s[i]=='{' and s[i-1]!='\\':level+=1
        elif s[i]=='}' and s[i-1]!='\\':level-=1
        i+=1
    return s[start+1:i-1],i


def unwrap(s,command):
    needle='\\'+command+'{'
    while needle in s:
        start=s.index(needle);body,end=braced(s,start+len(needle)-1)
        s=s[:start]+body+s[end:]
    return s


def clean(s):
    # Preserve TeX mathematics while converting surrounding prose to Markdown.
    mathematics=[]
    def save_math(match):
        mathematics.append(match[0])
        return 'MATHPLACEHOLDER'+str(len(mathematics)-1)+'END'
    s=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',save_math,s)
    s=s.replace('\\model{}','XFNet').replace('\\model','XFNet')
    s=s.replace('~',' ').replace('\\%','%').replace('\\_','_').replace('\\&','&')
    s=s.replace('---','—').replace('--','–').replace('``','“').replace("''",'”')
    s=re.sub(r'\\(?:\s+|\.)',' ',s)
    for cmd in ('textbf','mathbf','texttt','emph','text','operatorname','url'):
        needle='\\'+cmd+'{'
        while needle in s:
            start=s.index(needle);body,end=braced(s,start+len(needle)-1)
            body=('**'+body+'**') if cmd=='textbf' else ('`'+body+'`') if cmd in ('texttt','url') else body
            s=s[:start]+body+s[end:]
    for i,body in enumerate(mathematics):
        s=s.replace('MATHPLACEHOLDER'+str(i)+'END',body)
    return s


def run():
    tex=(ROOT/'main.tex').read_text(encoding='utf-8')
    refs={'sec:recent_comp':'4.2','sec:runtime':'4.6'};counters={'tab':0,'fig':0,'eq':0}
    for match in re.finditer(r'\\label\{([^}]+)\}',tex):
        key=match[1];kind=key.split(':',1)[0]
        if kind in counters:counters[kind]+=1;refs[key]=str(counters[kind])
    title=braced(tex,tex.index('\\title{')+6)[0]
    start=tex.index('\\begin{abstract}')
    s=tex[start:tex.index('\\bibliographystyle')]
    s=s.replace('\\begin{abstract}','## Abstract\n').replace('\\end{abstract}','\n')
    s=s.replace('\\noindent','').replace('\\linenumbers','')
    s=re.sub(r'\\ref\{([^}]+)\}',lambda m:refs.get(m[1],m[1]),s)
    s=re.sub(r'\\(subsection|section)(\*)?\{([^}]+)\}',lambda m:'\n'+('###' if m[1]=='subsection' else '##')+' '+m[3]+'\n',s)
    def table(m):
        block=m[0];cap=braced(block,block.index('\\caption{')+8)[0]
        num=re.search(r'\\label\{([^}]+)',block);number=refs[num[1]] if num else ''
        data=re.search(r'\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}',block,re.S)[1]
        data=re.sub(r'\\(?:toprule|midrule|bottomrule)','',data)
        data=re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{([^}]+)\}',r'\1',data)
        rows=[clean(r.strip().replace('\n',' ')) for r in data.split('\\\\') if r.strip()]
        width=max(len(r.split('&')) for r in rows)
        rows=[r.split('&')+['']*(width-len(r.split('&'))) for r in rows]
        body=['| '+' | '.join(c.strip() for c in row)+' |' for row in rows]
        body.insert(1,'| '+' | '.join(['---']*width)+' |')
        return '\n**Table '+number+'.** '+clean(cap)+'\n\n'+'\n'.join(body)+'\n'
    s=re.sub(r'\\begin\{table\}.*?\\end\{table\}',table,s,flags=re.S)
    def figure(m):
        block=m[0];cap=braced(block,block.index('\\caption{')+8)[0]
        filename=re.search(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',block)[1]
        filename=str(Path(filename).with_suffix('.svg')).replace('\\','/')
        lab=re.search(r'\\label\{([^}]+)',block);num=refs[lab[1]] if lab else ''
        return f'\n![Figure {num}]({filename})\n\n**Figure {num}.** '+clean(cap)+'\n'
    s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',figure,s,flags=re.S)
    s=re.sub(r'\\begin\{(?:equation|align)\}',lambda m:'\n$$\n'+('\\begin{aligned}\n' if 'align' in m[0] else ''),s)
    s=s.replace('\\end{equation}','\n$$\n').replace('\\end{align}','\n\\end{aligned}\n$$\n')
    s=re.sub(r'\\label\{[^}]+\}','',s)
    import json
    meta={r['key']:r for r in json.loads((ROOT/'analysis/references/crossref_metadata.json').read_text(encoding='utf-8'))}
    cited=[]
    def citation(m):
        names=[]
        for k in m[1].split(','):
            if k not in cited:cited.append(k)
            r=meta.get(k,{});authors=r.get('authors') or []
            if authors:
                author=authors[0].get('family','')
                if len(authors)>2:author+=' et al.'
                elif len(authors)==2:author+=' and '+authors[1].get('family','')
            else:author='Angelopoulos and Bates' if k=='angelopoulos2021gentle' else 'Vovk et al.' if k=='vovk2005algorithmic' else k
            year=re.search(r'\d{4}',k)
            names.append(author+(', '+year[0] if year else ''))
        return '('+'; '.join(names)+')'
    s=re.sub(r'\\citep\{([^}]+)\}',citation,s)
    s=clean(s)
    s=re.sub(r'\n{3,}','\n\n',s)
    header='# '+clean(title)+'\n\n'
    footer='\n\n## References\n\n'
    for k in sorted(cited):
        r=meta.get(k,{})
        if r.get('title'):
            authors=', '.join((a.get('family','')+', '+a.get('given','')) for a in (r.get('authors') or []))
            if not authors and k=='vovk2005algorithmic':authors='Vovk, Vladimir, Gammerman, Alex, and Shafer, Glenn'
            year=re.search(r'\d{4}',k)[0]
            footer+=f"- {authors} ({year}). {r['title'][0]}. https://doi.org/{r['doi']}\n"
        else:footer+='- Angelopoulos, A. N., and Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. https://doi.org/10.48550/arXiv.2107.07511\n'
    (ROOT/'MANUSCRIPT_EN.md').write_text(header+s.strip()+footer,encoding='utf-8')
    print(ROOT/'MANUSCRIPT_EN.md')


if __name__=='__main__':run()
