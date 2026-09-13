"""Publication-exportable visual audit of selected panels. No invented HBN results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(folder):
    root=Path(folder);out=root/'figures';out.mkdir(exist_ok=True)
    panels=json.loads((root/'selected_panels.json').read_text())
    paths=json.loads((root/'selection_paths.json').read_text())
    trials=json.loads((root/'candidate_evaluations.json').read_text())
    synthetic=json.loads((root/'environment.json').read_text())['synthetic']
    releases=[p['held_release'] for p in panels]
    all_groups=sorted(set(t['candidate_added'] for t in trials))
    if len(all_groups)>30:
        frequencies={g:sum(g in p['groups'] for p in panels) for g in all_groups}
        maximum={g:max(t['mean_inner_ba'] for t in trials if t['candidate_added']==g) for g in all_groups}
        all_groups=sorted(all_groups,key=lambda g:(-frequencies[g],-maximum[g],g))[:30]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig,name):
        if synthetic:fig.suptitle('SYNTHETIC SOFTWARE DEMO - NOT HBN RESULTS',fontsize=11,color='#a43723',weight='bold')
        fig.tight_layout(rect=(0,0,1,.94 if synthetic else 1))
        for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    matrix=np.array([[int(g in p['groups']) for p in panels] for g in all_groups])
    fig,ax=plt.subplots(figsize=(8,max(4,len(all_groups)*.27)))
    image=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(len(releases)),xticklabels=releases,yticks=range(len(all_groups)),yticklabels=all_groups,
           title='Selected panel by held-out release',xlabel='Outer test release')
    for i in range(len(all_groups)):
        for j in range(len(releases)):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j] else '#18324a')
    save(fig,'panel_selection_heatmap')
    co=matrix@matrix.T/len(panels)
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(co,cmap='YlGnBu',vmin=0,vmax=1)
    ax.set(xticks=range(len(all_groups)),xticklabels=all_groups,yticks=range(len(all_groups)),yticklabels=all_groups,
           title='Co-selection frequency (descriptive; not interaction strength)')
    plt.setp(ax.get_xticklabels(),rotation=65,ha='right');fig.colorbar(im,ax=ax,label='Fraction of outer training panels')
    save(fig,'co_selection_heatmap')
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,held in zip(axes.flat,releases):
        part=[t for t in trials if t['held_release']==held];steps=max(t['step'] for t in part)
        values=np.full((len(all_groups),steps),np.nan)
        path=[p for p in paths if p['held_release']==held]
        for t in part:
            if t['candidate_added'] not in all_groups:continue
            previous=.5 if t['step']==1 else path[t['step']-2]['mean_inner_ba']
            values[all_groups.index(t['candidate_added']),t['step']-1]=t['mean_inner_ba']-previous
        im=ax.imshow(np.ma.masked_invalid(values),cmap='RdBu',vmin=-.15,vmax=.15,aspect='auto')
        ax.set(title=held,xticks=range(steps),xticklabels=range(1,steps+1),xlabel='Forward search step',
               yticks=range(len(all_groups)),yticklabels=all_groups)
        for t in path:
            if t['candidate_added'] in all_groups:ax.plot(t['step']-1,all_groups.index(t['candidate_added']),'s',mfc='none',mec='#ec8d26',ms=13,mew=2)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Change in mean inner balanced accuracy',shrink=.7)
    # Explicit margins accommodate a shared colorbar without tight_layout warnings.
    if synthetic:fig.suptitle('SYNTHETIC SOFTWARE DEMO - NOT HBN RESULTS',color='#a43723',weight='bold')
    fig.subplots_adjust(left=.20,right=.79,wspace=.85,hspace=.35,top=.90,bottom=.08)
    for ext in ['pdf','png']:fig.savefig(out/f'conditional_gain_heatmap.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for panel in panels:
        path=[p for p in paths if p['held_release']==panel['held_release']]
        xx=[p['n_features'] for p in path];yy=[p['mean_inner_ba'] for p in path]
        ax.plot(xx,yy,'o-',label=panel['held_release'])
        ax.scatter(panel['n_features'],panel['mean_inner_ba'],s=150,facecolors='none',edgecolors='#111827',linewidth=1.5)
    ax.set(xlabel='Features in candidate panel',ylabel='Mean inner balanced accuracy',ylim=(0,1),
           title='Training-only selection paths; circles mark chosen panels');ax.legend(fontsize=8)
    save(fig,'selection_paths')
    scores=pd.read_csv(root/'metrics.csv');fig,ax=plt.subplots(figsize=(8,4.5))
    colors=['#18324a','#168b8b','#ea9534']
    for i,(model,color) in enumerate(zip(['first_step','selected','all_candidates'],colors)):
        part=scores[(scores.model==model)&(scores.release!='pooled')].set_index('release').loc[releases]
        ax.bar(np.arange(len(releases))+(i-1)*.24,part.balanced_accuracy,width=.23,label=model,color=color)
    ax.set(xticks=range(len(releases)),xticklabels=releases,ylim=(0,1),ylabel='Outer balanced accuracy',title='Untouched-release evaluation')
    ax.axhline(.5,ls='--',color='gray',lw=1);ax.legend(ncol=3,fontsize=8)
    save(fig,'held_out_performance')
    feature_names=sorted(set(f for p in panels for f in p['features']))
    counts={f:sum(f in p['features'] for p in panels) for f in feature_names}
    top=sorted(feature_names,key=lambda f:(-counts[f],f))[:30]
    values=np.array([[int(f in p['features']) for p in panels] for f in top])
    fig,ax=plt.subplots(figsize=(9,max(5,len(top)*.25)))
    ax.imshow(values,cmap='Greens',vmin=0,vmax=1,aspect='auto')
    ax.set(xticks=range(len(releases)),xticklabels=releases,yticks=range(len(top)),yticklabels=[f[3:] for f in top],
           title='Individual feature selection (up to 30 most frequent)')
    save(fig,'individual_feature_heatmap')
    jaccard=np.zeros((len(panels),len(panels)))
    for i,a in enumerate(panels):
        for j,b in enumerate(panels):
            aa,bb=set(a['features']),set(b['features']);jaccard[i,j]=len(aa&bb)/len(aa|bb)
    pd.DataFrame(jaccard,index=releases,columns=releases).to_csv(root/'panel_jaccard.csv')
    pd.DataFrame({'feature':feature_names,'outer_folds_selected':[counts[f] for f in feature_names]}).to_csv(root/'feature_frequency.csv',index=False)
    print(f'Saved 6 figure families as PDF and PNG in {out}')
