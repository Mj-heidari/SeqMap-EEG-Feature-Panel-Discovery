import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from hbn_r4.selection import candidate_groups,search_panel,fit_panel,predict_panel
from hbn_r4.evaluation import release_splits

class SequentialChecks(unittest.TestCase):
    def test_names_preserve_regions_and_feature_dimensions(self):
        names=['f__eo_sp_left_frontal_alpha_cf','f__ec_sp_left_temporal_theta','f__delta_co_A_B_theta','f__eo_gr_A_theta_strength']
        groups=candidate_groups(names)
        self.assertEqual(len(groups),4)
        self.assertIn('spectral_left_frontal',groups);self.assertIn('spectral_left_temporal',groups)
        self.assertEqual(sum(map(len,groups.values())),4)

    def test_forward_search_recovers_predictive_feature(self):
        rng=np.random.default_rng(42);y=np.tile([0,1],60)
        X=pd.DataFrame({'f__eo_sp_LF_theta':y*5+rng.normal(0,.1,len(y)),
                        'f__eo_sp_RF_theta':rng.normal(size=len(y))})
        cv=list(StratifiedKFold(3,shuffle=True,random_state=2).split(X,y))
        chosen,path,trials=search_panel(X,y,cv,candidate_groups(X.columns,'features'),{'max_steps':2,'n_estimators':20,'min_samples_leaf':2})
        self.assertEqual(path[0]['candidate_added'],'eo_sp_LF_theta')
        self.assertEqual(chosen['n_features'],1);self.assertEqual(len(trials),3)

    def test_imputer_frozen_on_hostile_test_data(self):
        X=pd.DataFrame({'f__a':[1.,np.nan,2.,3.,4.,5.,6.,7.]});y=np.tile([0,1],4)
        model=fit_panel(X,y,['f__a'],{'n_estimators':8})
        med=model[0].statistics_.copy();predict_panel(model,pd.DataFrame({'f__a':[np.nan,1e8]}),['f__a'])
        np.testing.assert_array_equal(med,model[0].statistics_)

    def test_family_blocking_spans_releases(self):
        X=pd.DataFrame({'release':['a','a','b','b'],'group_id':['f','g','f','h']})
        for held,train,test in release_splits(X):
            self.assertFalse(set(X.iloc[train].group_id)&set(X.iloc[test].group_id))
