"""Build a paragraph-aligned Chinese review copy from the English master.

Translations are author-review material, not an independent result source.
Paragraph identifiers and English hashes make later revisions detectable.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1] / 'outputs/paper_ps'

HEADINGS = {
    'SeisFormer: compact time–frequency fusion for P- and S-wave arrival picking across seismic networks':
        'XFNet：用于跨台网P、S波到时拾取的紧凑时频融合模型',
    'SeisFormer: compact time–frequency fusion for P- and S-wave arrival picking across multi-source seismic archives':
        'XFNet：用于多来源地震资料库P、S波到时拾取的紧凑时频融合模型',
    'XFNet: compact time–frequency fusion for P- and S-wave arrival picking across seismic networks':
        'XFNet：用于跨台网P、S波到时拾取的紧凑时频融合模型',
    'XFNet: compact time–frequency fusion for P- and S-wave arrival picking across multi-source seismic archives':
        'XFNet：用于多来源地震资料库P、S波到时拾取的紧凑时频融合模型',
    'Summary': '摘要', 'Abstract': '摘要', 'Introduction': '引言',
    'Relation to recent phase-picking studies': '与近期震相拾取研究的关系',
    'Waveform data and picking task': '波形资料与拾取任务',
    'Source inventory and physical standardization': '资料清单与物理标准化',
    'Event groups, crops and target availability': '事件分组、裁窗和目标可用性',
    'Model and estimation procedure': '模型与估计流程',
    'Compact temporal and spectral encoding': '紧凑时域与频谱编码',
    'Context, decoder and joint outputs': '上下文、解码器与联合输出',
    'Training and checkpoint selection': '训练与权重选择',
    'Picking and uncertainty': '拾取与不确定性',
    'Results': '结果',
    'Time–frequency contribution and training scale': '时频表示的贡献与训练规模',
    'Matched comparison with a recent dual-domain picker': '与近期双域拾取器的同预算比较',
    'Source-dependent timing and physical interpretation': '逐来源到时误差及物理解释',
    'Source-level intervals and confidence ranking': '逐来源区间与置信评分排序',
    'Held-out waveform examples': '留出测试波形示例',
    'Computational cost': '计算开销', 'Discussion': '讨论',
    'Implications for earthquake catalogue construction': '对地震目录构建的启示',
    'Arrival accuracy, representation trade-offs and downstream relevance': '到时精度、表示取舍与下游关联意义',
    'Uncertainty and transfer between collections': '不确定性与集合间迁移',
    'Scope and remaining comparisons': '适用范围与后续比较',
    'Conclusions': '结论', 'Data availability': '数据可用性',
    'Computer code and data availability': '计算机代码与数据可用性',
    'CRediT authorship contribution statement': 'CRediT 作者贡献声明',
    'Declaration of competing interest': '利益冲突声明',
    'Supporting information': '补充材料',
    'Supplementary data': '补充数据',
}

TABLE_CAPTIONS = [
    '具有项目分区的可用记录及主模型1M训练阶段的实际组成。事件/噪声属性沿用来源元数据。127,940条评价清单形成于CEED加入训练之前。',
    '262k表示受控实验。每行为使用相同训练记录与先导初始化的配对微调种子。SD为三个种子间的样本标准差。',
    'seed42在共同测试集上的规模实验。明确列出选模规则。阶段预算不包括此前热启动历史；MAE指验证平均P/S MAE，Det指验证检测F1。',
    '主权重在不同来源上的到时误差与拾取F1。数量为裁窗后已知目标；F1使用固定0.3秒容差和0.3震相评分阈值。NEIC因固定目标偏移单独解释。',
    '使用同一预测候选集合的长记录应用。参考匹配只借助参考到时选择已输出候选中的计分对象，用于诊断候选恢复；最强候选行不借参考到时选择。',
]

FIGURE_CAPTIONS = [
    'SeisFormer时频结构。时域主干与两条频谱路径形成375-token表示，供三个Transformer块处理。带跳跃连接的插值恢复输入采样间隔上的sigmoid P/S评分。独立头汇聚全部编码token完成窗口事件检测。无效样点在损失和取峰时屏蔽；主网络未使用掩码感知注意力。图中波形与输出峰为示意。',
    '（a）262k记录上配对时域/双域微调种子的到时误差。（b）验证MAE选模下的规模结果；空心菱形为独立按检测规则选出的主1M权重。1M阶段继续自524k模型，262k和524k阶段从先导模型开始。',
    'seed42时域对照、262k双域模型与主1M模型的逐来源P/S MAE。对数坐标展示清晰初至与误差长尾集合的区别。NEIC未绘入，因为常数预测器可复现其固定目标偏移。',
    '（a）同一主权重使用混合标定半径和逐来源标定半径时的经验覆盖；每点为一个来源—震相组，虚线为名义90%覆盖。（b）已知目标按震相峰值评分排序后的内部到时MAE。保留集合MAE下降同时伴随样本数减少，不代表全体样本精度。',
    '（a）每文件不借参考到时选择候选时的绝对误差分布。（b）这一规则与参考匹配的各相分数及数量扣分。总分在（b）中除以2，以共用1,000分坐标；两规则都保留完整输出数量惩罚。',
    '1,000记录应用中的最强候选波形实例。左：为显示而分别错开的标准化ENZ波形；虚线为参考P/S到时，实线为预测。右：同一记录以S到时为中心的窗口。显示振幅在99.5百分位截断以呈现弱运动，推理使用原标准化振幅。三行依次为准确、P错误、P准确/S错误分层内的中位误差样本。坐标范围外的S预测写在子图标题。文件身份与选例规则随图数据提供。',
]

CELL_TRANSLATIONS = {
    'Source': '来源', 'Available records': '可用记录', 'Training events': '训练事件',
    'Training noise': '训练噪声', 'Total': '总计', 'Seed/configuration': '种子/配置',
    'Temporal P': '时域P', 'Dual P': '双域P', 'Temporal S': '时域S', 'Dual S': '双域S',
    'Arrival MAE (s)': '到时MAE（秒）', 'Mean': '平均', 'SD': '标准差',
    'Training stage': '训练阶段', 'Rule': '选模', 'P MAE (s)': 'P MAE（秒）',
    'S MAE (s)': 'S MAE（秒）', 'Det F1': '检测F1', 'Noise FPR (%)': '噪声FPR（%）',
    '262k, 20 epochs': '262k，20轮', '524k, 20 epochs': '524k，20轮',
    '1M, 20 epochs': '1M，20轮', 'Det': 'Det', 'MAE': 'MAE',
    'P targets': 'P目标数', 'S targets': 'S目标数',
    'Candidate used for scoring': '计分候选选择', 'P /1000': 'P /1000', 'S /1000': 'S /1000',
    'Penalty': '扣分', 'Total /2000': '总分/2000',
    'Strongest, without reference': '最强，无参考', 'Reference matched': '参考匹配',
    'Model': '模型', 'Parameters': '参数量', 'GPU B1 (ms)': 'GPU B1（ms）',
    'GPU B64 (s$^{-1}$)': 'GPU B64（窗/秒）', 'CPU B1 (ms)': 'CPU B1（ms）',
    'Macro-mean (excl. NEIC)': '宏平均（不含NEIC）', 'Pooled (all known)': '合并微平均（全部已知）',
}


def units_from_markdown(text):
    body = text.split('\n## References\n')[0]
    units = []; section = ''; paragraph = 0
    for block in re.split(r'\n\s*\n', body.strip()):
        block = block.strip()
        if not block: continue
        # Display math is joined before splitting, below.
        if block.startswith('#'):
            kind = 'heading'; section = block.lstrip('# ').strip()
        elif block.startswith('**Key words:') or block.startswith('**Keywords:'): kind = 'keywords'
        elif block.startswith('**Table '): kind = 'table_caption'
        elif block.startswith('|'): kind = 'table'
        elif block.startswith('![Figure '): kind = 'figure'
        elif block.startswith('**Figure '): kind = 'figure_caption'
        elif block.startswith('DISPLAYMATHTOKEN'):
            kind = 'math'; block = MATH_BLOCKS[int(re.search(r'\d+', block)[0])]
        else: kind = 'paragraph'
        unit = dict(kind=kind, section=section, english=block)
        if kind == 'paragraph':
            paragraph += 1; unit['id'] = f'P{paragraph:03d}'
        units.append(unit)
    return units


def protect_display_math(text):
    global MATH_BLOCKS
    MATH_BLOCKS = []
    def save(m):
        MATH_BLOCKS.append(m[0]); return '\n\nDISPLAYMATHTOKEN'+str(len(MATH_BLOCKS)-1)+'\n\n'
    return re.sub(r'\$\$[\s\S]*?\$\$', save, text)


def citations(english):
    return [m[0] for m in re.finditer(r'\([^()]*[A-Za-zÀ-ž][^()]*\b(?:19|20)\d{2}[a-z]?[^()]*\)', english)]


def translate_table(text):
    lines = text.splitlines(); result = []
    for i, line in enumerate(lines):
        cells = [v.strip() for v in line.strip('|').split('|')]
        if i == 1:
            result.append('| '+' | '.join(['---']*len(cells))+' |'); continue
        result.append('| '+' | '.join(CELL_TRANSLATIONS.get(v, v) for v in cells)+' |')
    return '\n'.join(result)


def tex_prose(text):
    maths = []
    def save(m):
        maths.append(m[0]); return 'TEXMATH'+str(len(maths)-1)+'END'
    text = re.sub(r'\$[^$]+\$', save, text)
    text = text.replace('\\', r'\textbackslash{}')
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]: text = text.replace(a,b)
    text = re.sub(r'\*\*(.+?)\*\*',lambda m:r'\textbf{'+m[1]+'}',text)
    for i, math in enumerate(maths): text = text.replace('TEXMATH'+str(i)+'END', math)
    return text


def tex_table(text, caption):
    rows = [[v.strip() for v in line.strip('|').split('|')] for line in text.splitlines()]
    rows.pop(1); n = len(rows[0])
    # A wide label column and equal-width numeric columns keep the tables on-page.
    spec = r'>{\raggedright\arraybackslash}p{'+('35mm' if n<=6 else '22mm')+'}'+r'*{'+str(n-1)+r'}{>{\centering\arraybackslash}X}'
    result = [r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+tex_prose(caption)+'}',
              r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule']
    for i,row in enumerate(rows):
        result.append(' & '.join(tex_prose(v) for v in row)+r'\\')
        if i==0: result.append(r'\midrule')
    result += [r'\bottomrule',r'\end{tabularx}',r'\end{table}']
    return '\n'.join(result)


def run():
    source = (ROOT/'MANUSCRIPT_EN.md').read_text(encoding='utf-8')
    units = units_from_markdown(protect_display_math(source))
    translations = json.loads((ROOT/'analysis/review_translation_zh.json').read_text(encoding='utf-8'))
    paragraphs = [u for u in units if u['kind']=='paragraph']
    assert len(paragraphs)==len(translations) and {u['id'] for u in paragraphs}==set(translations)
    labels_path=ROOT/'analysis/review_labels_zh.json'
    labels=json.loads(labels_path.read_text(encoding='utf-8')) if labels_path.exists() else {}
    table_captions=labels.get('table_captions',TABLE_CAPTIONS)
    figure_captions=labels.get('figure_captions',FIGURE_CAPTIONS)
    expected_tables=sum(u['kind']=='table' for u in units)
    expected_figures=sum(u['kind']=='figure' for u in units)
    expected_equations=sum(u['kind']=='math' for u in units)
    assert len(table_captions)==expected_tables and len(figure_captions)==expected_figures
    # Use the untouched TeX equations rather than a prose-cleaned approximation.
    main = (ROOT/'main.tex').read_text(encoding='utf-8')
    equations = [m[2] for m in re.finditer(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}', main, re.S)]
    assert len(equations)==expected_equations
    equations = [re.sub(r'\\label\{[^}]+\}', '', v).strip() for v in equations]
    notes = (f'核对说明：本稿与英文主文的{len(paragraphs)}个正文段落逐段对应，保留{expected_equations}组公式、{expected_tables}张表和{expected_figures}张图。'
             f'P001—P{len(paragraphs):03d}用于定位修改；图内英文标注与英文稿共用，参考文献沿用英文条目。'
             '本文的评价对象为P/S家族到时与窗口事件检测。')
    zh = ['> '+notes]; bilingual = ['# XFNet中英逐段核对','> '+notes]
    tex = [r'''\documentclass[a4paper,11pt]{article}
\usepackage{fontspec,xeCJK}
\setmainfont{Times New Roman}
\setCJKmainfont{SimSun}[BoldFont=SimHei,ItalicFont=KaiTi]
\setCJKsansfont{Microsoft YaHei}
\setCJKmonofont{SimSun}
\usepackage[margin=23mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,array,caption,placeins,xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,bookmarksnumbered=true}
\newcommand{\bce}{\operatorname{BCE}}
\captionsetup{font=small,labelfont=bf}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\topfraction}{0.9}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\setlength{\emergencystretch}{3em}
\linespread{1.15}
\title{XFNet：用于多来源地震资料库P、S波到时拾取的紧凑时频融合模型}
\author{向巍$^{1,2,*}$，牟剑英$^{1}$（中文逐段核对版）\\[2pt]
\parbox{0.92\textwidth}{\centering\small
$^1$广西壮族自治区地震局，南宁 530022\\[1pt]
$^2$广西北海涠洲岛海洋地球物理国家野外科学观测研究站/涠洲岛地震监测中心站，北海 536000\\[1pt]
$^*$通讯作者：向巍（\texttt{xiangwei@gxearthquake.cn}；ORCID：\texttt{0009-0001-4572-3507}）}}
\date{2026年10月24日}
\begin{document}
\maketitle
''',r'{\small\color{gray}'+tex_prose(notes)+r'}\par\medskip']
    aligned = []; table_idx = 0; figure_idx = 0; equation_idx = 0; pending_caption = ''
    for unit in units:
        kind = unit['kind']; english = unit['english']
        if kind=='heading':
            title = HEADINGS[unit['section']]; level = len(re.match(r'#+',english)[0])
            if level==1:
                zh.insert(0, '# '+title); continue
            chinese = '#'*level+' '+title
            tex.append(('\\section*' if level==2 and title in ('摘要','数据可用性','计算机代码与数据可用性','CRediT 作者贡献声明','利益冲突声明','补充材料','补充数据') else '\\section' if level==2 else '\\subsection')+'{'+title+'}')
        elif kind=='paragraph':
            pid = unit['id']; text = translations[pid]
            refs = citations(english)
            if refs: text += ' '+' '.join(refs)
            chinese = '**['+pid+']** '+text
            bilingual += ['## '+pid+' · '+HEADINGS[unit['section']], '**English**\n\n'+english, '**中文**\n\n'+text]
            tex.append(r'\noindent{\scriptsize\color{gray}['+pid+']} '+tex_prose(text)+r'\par\medskip')
            aligned.append(dict(id=pid,section_en=unit['section'],section_zh=HEADINGS[unit['section']],
                english=english,chinese=text,english_sha256=hashlib.sha256(english.encode()).hexdigest()))
        elif kind=='keywords':
            chinese='**关键词：** 震相拾取；时频特征融合；Transformer；共形预测；地震事件检测。'
            tex.append(tex_prose(chinese)+r'\par\medskip')
        elif kind=='table_caption':
            pending_caption = table_captions[table_idx]
            chinese=f'**表{table_idx+1}：** '+pending_caption
        elif kind=='table':
            chinese=translate_table(english)
            tex.append(tex_table(chinese,pending_caption)); table_idx+=1
        elif kind=='figure':
            path=re.search(r'\]\(([^)]+)\)',english)[1]
            chinese=f'![图{figure_idx+1}]({path})'
            pdf=str(Path(path).with_suffix('.pdf')).replace('\\','/')
            tex.append(r'\begin{figure}[htbp]\centering'+ '\n'+r'\includegraphics[width=\linewidth]{'+pdf+'}')
        elif kind=='figure_caption':
            caption=figure_captions[figure_idx]
            chinese=f'**图{figure_idx+1}：** '+caption
            tex.append(r'\caption{'+tex_prose(caption)+r'}\end{figure}')
            figure_idx+=1
        elif kind=='math':
            equation=equations[equation_idx]
            # The aligned loss needs an aligned environment inside display math.
            if equation_idx==2: equation='\\begin{aligned}\n'+equation+'\n\\end{aligned}'
            chinese='$$\n'+equation+'\n$$'
            tex.append('\\[\n'+equation+'\n\\]')
            equation_idx+=1
        else: raise ValueError(kind)
        zh.append(chinese)
        if kind!='paragraph': bilingual.append(chinese)
    assert (table_idx,figure_idx,equation_idx)==(expected_tables,expected_figures,expected_equations)
    refs=source.split('\n## References\n',1)[1]
    zh += ['## 参考文献（与英文稿一致）',refs.strip()]
    bilingual += ['## 参考文献（与英文稿一致）',refs.strip()]
    glossary=('事件检测F1评价窗口内有无事件；拾取F1评价相位是否在指定容差内命中。'
              'MAE为平均绝对误差，不能与其他论文的残差标准差直接比较。'
              '峰值评分用于排序，不自动具有校准概率含义。'
              '时域/双域表示消融的三种子共享先导初始化；近期模型比较的三种子各自从零训练。'
              '标定平均覆盖偏差与最大覆盖偏差分别为不同统计量。')
    zh += ['## 核对用术语说明',glossary]
    tex += [r'\FloatBarrier\section*{核对用术语说明}',tex_prose(glossary),
            r'\section*{参考文献（与英文稿一致）}\small']
    for line in refs.strip().splitlines():
        if not line.startswith('- '): continue
        title, sep, doi=line[2:].rpartition(' https://doi.org/')
        if sep:
            tex.append(r'\noindent '+tex_prose(title)+r' \url{https://doi.org/'+doi+r'}\par\medskip')
        else: tex.append(tex_prose(line[2:])+r'\par\medskip')
    tex.append(r'\end{document}')
    (ROOT/'中文核对版.md').write_text('\n\n'.join(zh)+'\n',encoding='utf-8')
    (ROOT/'中英逐段对照.md').write_text('\n\n'.join(bilingual)+'\n',encoding='utf-8')
    (ROOT/'chinese_review.tex').write_text('\n\n'.join(tex)+'\n',encoding='utf-8')
    (ROOT/'analysis/review_units.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'analysis/bilingual_alignment.json').write_text(json.dumps(dict(
        english_master_sha256=hashlib.sha256(main.encode()).hexdigest(),
        paragraphs=aligned,tables=table_idx,figures=figure_idx,equations=equation_idx),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Chinese checking copy: {len(paragraphs)} aligned paragraphs, {table_idx} tables, {figure_idx} figures, {equation_idx} equation groups.')


if __name__=='__main__': run()
