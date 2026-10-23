"""Publication figures tied to the preserved P/S checkpoint and saved outputs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/paper_ps')
F=ROOT/'figures'
BLUE='#347ca8';RED='#ba6170';GREEN='#508e7b';GRAY='#7a8794'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'savefig.dpi':240})


def save(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(F/(name+'.'+ext),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def run():
    e=json.loads((ROOT/'analysis/submission_evidence.json').read_text(encoding='utf-8'))
    # Preserve the user's abstract tensor/stack architecture illustration, with
    # its pooling annotation corrected to match the implementation.
    svg=(F/'model_architecture_en.svg').read_text(encoding='utf-8')
    svg=svg.replace('Valid-Token Mean Pooling','Token Mean Pooling')
    svg=svg.replace('Phase Probability Curves','Phase Score Curves')
    (F/'ps_architecture.svg').write_text(svg,encoding='utf-8')
    import cairosvg
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(F/'ps_architecture.pdf'))
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(F/'ps_architecture.png'),output_width=2000)
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.75),layout='constrained')
    for j,p in enumerate(('P','S')):
        for row in e['controlled_seeds']:
            a,b=row['time'][p],row['dual'][p]
            ax[0].plot([j*3,j*3+1],[a,b],'-o',ms=4,lw=1,color=BLUE if j==0 else RED,alpha=.75)
        ax[0].text(j*3+.5,.62 if p=='P' else .42,p+' wave',ha='center',fontweight='bold')
    ax[0].set_xticks([0,1,3,4],['Time','Dual','Time','Dual']);ax[0].set_ylabel('Arrival MAE (s)');ax[0].set_ylim(.27,.65)
    ax[0].set_title('(a) Paired fine-tuning seeds, 262k rows',loc='left')
    xx=[262144,524288,1048576]
    for p,color in (('P',BLUE),('S',RED)):
        y=[e['runs'][n]['eval_test']['overall'][p]['mae_s'] for n in ('standardized_dual_262k','standardized_dual_expanded_524k','standardized_dual_1m_20ep')]
        ax[1].plot(xx,y,'o-',color=color,label=p+'; validation MAE selection')
        yy=e['primary']['overall'][p]['mae_s'];ax[1].scatter([xx[-1]],[yy],marker='D',s=32,facecolors='white',edgecolors=color,zorder=4)
    ax[1].set_xscale('log',base=2);ax[1].set_xticks(xx,['262k','524k','1M']);ax[1].set_xlabel('Training rows');ax[1].set_ylabel('Arrival MAE (s)')
    ax[1].set_title('(b) Training population and continuation',loc='left');ax[1].legend(frameon=False,fontsize=7)
    ax[1].text(.05,.08,'Open diamonds: 1M detection-selected',transform=ax[1].transAxes,fontsize=7)
    save(fig,'ps_control_and_scaling')
    sources=['STEAD','TXED','PNW','ETHZ','IQUIQUE','GEOFON','CREW']
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.15),layout='constrained')
    for j,(p,color) in enumerate((('P',BLUE),('S',RED))):
        a=axes[j]
        for run,marker,line,alpha in [('standardized_time_ondual_262k','s','--',.6),('standardized_dual_262k','o',':',.65),('standardized_dual_1m_20ep','D','-',1)]:
            item=e['runs'][run]['eval_test']['by_source'] if run!='standardized_dual_1m_20ep' else e['primary']['by_source']
            vals=[item[s][p]['mae_s'] for s in sources]
            a.plot(np.arange(len(sources)),vals,marker+line,color=color,alpha=alpha,ms=4,lw=1,
                label={'standardized_time_ondual_262k':'Time, 262k','standardized_dual_262k':'Dual, 262k','standardized_dual_1m_20ep':'Dual, 1M primary'}[run])
        a.set_xticks(range(len(sources)),sources,rotation=40,ha='right');a.set_yscale('log');a.set_ylabel('Arrival MAE (s)')
        a.set_title(f'({chr(97+j)}) {p} arrivals',loc='left');a.grid(axis='y',alpha=.2);a.legend(frameon=False,fontsize=7)
    save(fig,'ps_source_errors')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.5),layout='constrained')
    rows=sorted(e['primary_source_calibration'],key=lambda v:(v['source'],v['phase']))
    labels=[r['source']+' '+r['phase'] for r in rows];yy=np.arange(len(rows))
    for i,r in enumerate(rows):axes[0].plot([r['global_coverage'],r['local_coverage']],[i,i],color='#bac3ca',lw=1)
    axes[0].scatter([r['global_coverage'] for r in rows],yy,s=18,color=GRAY,label='Pooled radius')
    axes[0].scatter([r['local_coverage'] for r in rows],yy,s=18,color=GREEN,label='Source-specific radius')
    axes[0].axvline(.9,color='black',ls='--',lw=.8);axes[0].set_yticks(yy,labels,fontsize=7);axes[0].invert_yaxis();axes[0].set_xlim(.38,1.02)
    axes[0].set_xlabel('Empirical coverage');axes[0].set_title('(a) Same primary checkpoint, nominal 90%',loc='left');axes[0].legend(frameon=False,fontsize=7,loc='lower left')
    for p,col in (('P',BLUE),('S',RED)):
        rr=[r for r in e['selective_primary'] if r['phase']==p]
        axes[1].plot([r['fraction'] for r in rr],[r['mae_s'] for r in rr],'o-',color=col,label=p)
    axes[1].set_xlabel('Retained fraction by phase-peak confidence');axes[1].set_ylabel('MAE on retained arrivals (s)')
    axes[1].set_title('(b) Internal confidence-ranked residuals',loc='left');axes[1].legend(frameon=False);axes[1].grid(alpha=.2)
    save(fig,'ps_calibration_and_selection')
    from scripts.make_ps_heldout_waveforms import run as heldout
    heldout()
    print('Generated five figure sets from preserved multi-source test evidence')


if __name__=='__main__':run()
