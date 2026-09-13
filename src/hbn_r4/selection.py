"""Sequential forward selection with a parsimonious one-SE panel rule.

The outer test release is never passed to search_panel. Candidate features are
prespecified by names, not supervised ranking over held-out participants.
"""
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def candidate_groups(columns,mode='blocks'):
    names=sorted(c for c in columns if c.startswith('f__'))
    if mode=='bands':
        names=[c for c in names if not c.startswith('f__delta_') and '_sp_' in c and c.rsplit('_',1)[-1] in ('theta','alpha','beta')]
        return {n[3:]:[n] for n in names}
    if mode=='features':return {n[3:]:[n] for n in names}
    if mode!='blocks':raise ValueError('candidates must be blocks, bands, or features')
    groups={}
    for name in names:
        short=name[3:];condition,kind,tail=short.split('_',2)
        if kind=='sp':
            match=re.fullmatch(r'(.*)_(theta|alpha|beta|exponent|offset|alpha_cf|alpha_height)',tail)
            if not match:raise ValueError(f'Unknown spectral feature: {name}')
            group='spectral_'+match.group(1)
        elif kind=='co':group='connectivity_'+tail.rsplit('_',1)[-1]
        elif kind=='gr':group='strength_'+tail.rsplit('_',2)[-2]
        else:raise ValueError(f'Unknown feature naming convention: {name}')
        groups.setdefault(group,[]).append(name)
    return groups


def fit_panel(frame,y,columns,config):
    # Every call refits medians on this training partition only.
    values=frame[columns].to_numpy(float)
    values=np.where(np.isfinite(values),values,np.nan)
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
        RandomForestClassifier(n_estimators=config.get('n_estimators',200),
          min_samples_leaf=config.get('min_samples_leaf',3),max_depth=config.get('max_depth',None),
          max_features='sqrt',class_weight='balanced',random_state=config.get('seed',2026),n_jobs=1))
    model.fit(values,y)
    return model


def predict_panel(model,frame,columns):
    x=frame[columns].to_numpy(float);x=np.where(np.isfinite(x),x,np.nan)
    return model.predict_proba(x)[:,1]


def search_panel(frame,y,inner,groups,config):
    from sklearn.metrics import balanced_accuracy_score
    if not groups:raise ValueError('No candidate groups match the selected mode')
    max_steps=min(int(config.get('max_steps',6)),len(groups))
    if max_steps<1:raise ValueError('max_steps must be positive')
    chosen=[];trajectory=[];evaluations=[]
    for step in range(1,max_steps+1):
        candidates=[]
        for group in sorted(set(groups)-set(chosen)):
            columns=[c for g in chosen+[group] for c in groups[g]]
            scores=[]
            for train,val in inner:
                model=fit_panel(frame.iloc[train],y[train],columns,config)
                probability=predict_panel(model,frame.iloc[val],columns)
                scores.append(float(balanced_accuracy_score(y[val],probability>=.5)))
            mean=float(np.mean(scores));se=float(np.std(scores,ddof=1)/np.sqrt(len(scores))) if len(scores)>1 else 0.
            entry={'step':step,'candidate_added':group,'groups':chosen+[group],
                'features':columns,'n_features':len(columns),'mean_inner_ba':mean,'heuristic_se':se,'fold_scores':scores}
            candidates.append(entry);evaluations.append(entry)
        # Deterministic ties: fewer features then lexical group name.
        winner=min(candidates,key=lambda a:(-a['mean_inner_ba'],a['n_features'],a['candidate_added']))
        chosen=winner['groups'];trajectory.append(winner)
        print(f"  step {step}: +{winner['candidate_added']}; inner BA={winner['mean_inner_ba']:.3f}",flush=True)
    peak=max(trajectory,key=lambda a:a['mean_inner_ba'])
    tolerance=peak['heuristic_se'] if config.get('one_se_rule',True) else 0.
    eligible=[a for a in trajectory if a['mean_inner_ba']>=peak['mean_inner_ba']-tolerance]
    selected=min(eligible,key=lambda a:(a['n_features'],a['step']))
    return selected,trajectory,evaluations
