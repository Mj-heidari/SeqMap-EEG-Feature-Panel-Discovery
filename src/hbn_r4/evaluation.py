from pathlib import Path
import json
import hashlib
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
from .selection import candidate_groups,search_panel,fit_panel,predict_panel


def release_splits(table):
    for release in sorted(table.release.unique()):
        test=np.flatnonzero(table.release.to_numpy()==release)
        family=set(table.iloc[test].group_id)
        train=np.flatnonzero((table.release.to_numpy()!=release)&~table.group_id.isin(family).to_numpy())
        if not len(train):raise ValueError('No training participants after family blocking')
        yield release,train,test


def metric(rows):
    y=rows.label.to_numpy(int);p=rows.probability.to_numpy();pred=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(rows),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def run(table,config,outdir):
    out=Path(outdir)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose an empty result directory')
    required=['participant_id','group_id','release','label','duration']
    if not set(required).issubset(table) or table[required].isna().any().any():raise ValueError('Incomplete feature metadata')
    if not table.label.isin([0,1]).all():raise ValueError('Labels must be binary 0/1')
    table=table[table.duration==config.get('duration',60)].reset_index(drop=True)
    if table.empty or table.participant_id.duplicated().any():raise ValueError('Need unique participants at selected duration')
    from hbn_r3.data import RELEASES
    if set(table.release)!=set(RELEASES):raise ValueError('Primary study requires all four release labels')
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','matplotlib','joblib']},
        'feature_table_sha256':hashlib.sha256(pd.util.hash_pandas_object(table,index=True).values.tobytes()).hexdigest(),
        'synthetic':bool(table.participant_id.astype(str).str.startswith('SYNTH_').any())},indent=2))
    groups=candidate_groups(table.columns,config.get('candidates','blocks'));predictions=[];panels=[];paths=[];all_trials=[];splitlog=[]
    for held,train,test in release_splits(table):
        tr=table.iloc[train].reset_index(drop=True);te=table.iloc[test];y=tr.label.to_numpy(int)
        if tr.label.nunique()!=2 or te.label.nunique()!=2:raise ValueError('Every outer partition needs both classes')
        cv=StratifiedGroupKFold(config.get('inner_folds',3),shuffle=True,random_state=config.get('seed',2026))
        inner=list(cv.split(tr,tr.release+'|'+tr.label.astype(str),groups=tr.group_id))
        for it,iv in inner:
            if set(tr.iloc[it].group_id)&set(tr.iloc[iv].group_id):raise ValueError('Family leakage')
            if tr.iloc[it].label.nunique()!=2 or tr.iloc[iv].label.nunique()!=2:raise ValueError('Inner fold lacks class')
        print(f'Selecting panel with {held} held out',flush=True)
        selected,path,trials=search_panel(tr,y,inner,groups,config)
        panels.append({'held_release':held,**selected})
        paths.extend({'held_release':held,**entry} for entry in path)
        all_trials.extend({'held_release':held,**entry} for entry in trials)
        splitlog.append({'held_release':held,'train_ids':tr.participant_id.tolist(),'test_ids':te.participant_id.tolist(),
            'inner':[{'train_ids':tr.iloc[it].participant_id.tolist(),'validation_ids':tr.iloc[iv].participant_id.tolist()} for it,iv in inner]})
        # Benchmark against all candidate features and the first selected block/feature.
        variants={'selected':selected['features'],'all_candidates':[c for v in groups.values() for c in v],
            'first_step':path[0]['features']}
        for name,columns in variants.items():
            model=fit_panel(tr,y,columns,config);probability=predict_panel(model,te,columns)
            joblib.dump({'model':model,'columns':columns},out/f'model_{held}_{name}.joblib')
            rows=te[['participant_id','group_id','release','label']].copy();rows['probability']=probability
            rows['model']=name;rows['n_features']=len(columns);predictions.append(rows)
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(out/'oof_predictions.csv',index=False)
    for filename,content in [('selected_panels',panels),('selection_paths',paths),('candidate_evaluations',all_trials),('splits',splitlog)]:
        (out/(filename+'.json')).write_text(json.dumps(content,indent=2))
    reports=[]
    for model,rows in pred.groupby('model'):
        reports.append({'model':model,'release':'pooled',**metric(rows)})
        for release,part in rows.groupby('release'):reports.append({'model':model,'release':release,**metric(part)})
    pd.DataFrame(reports).to_csv(out/'metrics.csv',index=False)
    # Pair family-cluster resamples of already fixed OOF predictions.
    selected=pred[pred.model=='selected'];comparisons=[];rng=np.random.default_rng(config.get('seed',2026))
    for baseline in ['all_candidates','first_step']:
        other=pred[pred.model==baseline]
        merged=selected.merge(other,on=['participant_id','group_id','release','label'],suffixes=('_s','_b'),validate='one_to_one')
        family=merged.group_id.unique();pieces={g:merged[merged.group_id==g] for g in family};deltas=[]
        def delta(frame):return float(balanced_accuracy_score(frame.label,frame.probability_s>=.5)-balanced_accuracy_score(frame.label,frame.probability_b>=.5))
        for _ in range(config.get('bootstrap',1000)):
            sample=pd.concat([pieces[g] for g in rng.choice(family,len(family),replace=True)])
            if sample.label.nunique()==2:deltas.append(delta(sample))
        comparisons.append({'baseline':baseline,'delta_ba':delta(merged),
            'conditional_family_ci95':np.quantile(deltas,[.025,.975]).tolist() if deltas else [None,None]})
    (out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    return pred
