import os
import numpy as np              
import pandas as pd
import matplotlib.pyplot as plt 

import sys
import time
import math
import json
import itertools
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, model_selection, metrics, neighbors, ensemble, feature_selection
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture
import optuna
import optuna.visualization as ov

ann_path = '../emomusic/annotations/'
mod_path = '../models/'
graph_path = '../graphs/'
clips_path = '../emomusic/clips/'
random_state = 333

#Getting train and test indices
df = pd.read_csv(ann_path+'songs_info.csv')
df = df[['song_id', 'Mediaeval 2013 set']]

song_id_train = list(df[df['Mediaeval 2013 set'] == 'development']['song_id'])
train_idx = []
for song_id in song_id_train:
    for sam in range(7500, 45001, 2500):
        train_idx.append(str(song_id)+'_'+str(sam))

song_id_test = list(df[df['Mediaeval 2013 set'] == 'evaluation']['song_id'])
test_idx = []
for song_id in song_id_test:
    for sam in range(7500, 45001, 2500):
        test_idx.append(str(song_id)+'_'+str(sam))

#Loading data for trainning and testing
cont = pd.read_parquet(ann_path+'cont_selected_features_5+25.pqt')
diff = cont.diff(axis=0)
to_drop = [i for i in cont.index if i.split('_')[1] == '5000']
diff = diff.drop(to_drop, axis=0)

x = diff.drop(['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std'], axis=1)
ar_mean = diff['arousal_mean']
ar_std = diff['arousal_std']
va_mean = diff['valence_mean']
va_std = diff['valence_std']

'''x = cont.drop(['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std'], axis=1)
ar_mean = cont['arousal_mean']
ar_std = cont['arousal_std']
va_mean = cont['valence_mean']
va_std = cont['valence_std']'''

x_train = x.loc[train_idx]
ar_mean_train = ar_mean.loc[train_idx]
ar_std_train = ar_std.loc[train_idx]
va_mean_train = va_mean.loc[train_idx]
va_std_train = va_std.loc[train_idx]

x_test = x.loc[test_idx]
ar_mean_test = ar_mean.loc[test_idx]
ar_std_test = ar_std.loc[test_idx]
va_mean_test = va_mean.loc[test_idx]
va_std_test = va_std.loc[test_idx]

#Getting final labels
#y = cont[['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std']]
y = diff[['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std']]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]


###For static information
#Loading data for trainning and testing
stat = pd.read_parquet(ann_path+'static_selected_features.pqt')
x_stat = stat.drop(['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std'], axis=1)
ar_mean_stat = stat['arousal_mean']
ar_std_stat = stat['arousal_std']
va_mean_stat = stat['valence_mean']
va_std_stat = stat['valence_std']

x_train_stat = x_stat.loc[song_id_train]
ar_mean_train_stat = ar_mean_stat.loc[song_id_train]
ar_std_train_stat = ar_std_stat.loc[song_id_train]
va_mean_train_stat = va_mean_stat.loc[song_id_train]
va_std_train_stat = va_std_stat.loc[song_id_train]

x_test_stat = x_stat.loc[song_id_test]
ar_mean_test_stat = ar_mean_stat.loc[song_id_test]
ar_std_test_stat = ar_std_stat.loc[song_id_test]
va_mean_test_stat = va_mean_stat.loc[song_id_test]
va_std_test_stat = va_std_stat.loc[song_id_test]

train_sets = [('arousal_mean', ar_mean_train_stat, ar_mean_test_stat), ('arousal_std', ar_std_train_stat, ar_std_test_stat), ('valence_mean', va_mean_train_stat, va_mean_test_stat), ('valence_std', va_std_train_stat, va_std_test_stat)]

#models information
with open('models', 'r') as f:
    models = json.load(f)

def ran(l):
    return np.ptp(l)

#############################
## GAUSSIAN MIXTURE MODELS ##
#############################

## GETTING BEST K
'''
K_list = range(1,10)
BIC_list = [] 

#Best K for features
for k in K_list:
    gmm = mixture.GaussianMixture(n_components=k,covariance_type='full').fit(x_train)
    BIC_list.append(gmm.bic(x_train))
    
plt.plot(K_list,BIC_list)
plt.xlabel('Comp. feat')
plt.ylabel('BIC')
plt.grid()
plt.show() 

#Best K for labels
BIC_list = [] 
for k in K_list:
    gmm = mixture.GaussianMixture(n_components=k,covariance_type='full').fit(y_train)
    BIC_list.append(gmm.bic(y_train))
    
plt.plot(K_list,BIC_list)
plt.xlabel('Comp. labels')
plt.ylabel('BIC')
plt.grid()
plt.show()'''

##GETTING BEST COVARIANCE TYPE AND K
'''
NMI_spherical = []
NMI_full = []
NMI_diag = []
K_list = range(2,10)

for k in K_list:
    gmm_feat = mixture.GaussianMixture(n_components=k,covariance_type='spherical').fit(x_train)
    class_feat = gmm_feat.predict(x_train)
    gmm_lab = mixture.GaussianMixture(n_components=k,covariance_type='spherical').fit(y_train)
    class_lab = gmm_lab.predict(y_train)
    NMI_spherical.append(metrics.normalized_mutual_info_score(class_lab, class_feat))
        
    gmm_feat = mixture.GaussianMixture(n_components=k,covariance_type='full').fit(x_train)
    class_feat = gmm_feat.predict(x_train)
    gmm_lab = mixture.GaussianMixture(n_components=k,covariance_type='full').fit(y_train)
    class_lab = gmm_lab.predict(y_train)
    NMI_diag.append(metrics.normalized_mutual_info_score(class_lab, class_feat))

    gmm_feat = mixture.GaussianMixture(n_components=k,covariance_type='diag').fit(x_train)
    class_feat = gmm_feat.predict(x_train)
    gmm_lab = mixture.GaussianMixture(n_components=k,covariance_type='diag').fit(y_train)
    class_lab = gmm_lab.predict(y_train)
    NMI_full.append(metrics.normalized_mutual_info_score(class_lab, class_feat))
    
plt.plot(K_list,NMI_spherical,label='cov spherical')
plt.plot(K_list,NMI_full,label='cov full')
plt.plot(K_list,NMI_diag,label='cov diag')
plt.xlabel('Number of components')
plt.ylabel('NMI')
plt.legend()
plt.grid()
plt.show()
'''

k=4
cov_type = 'diag'
gmm_feat = mixture.GaussianMixture(n_components=k,covariance_type=cov_type).fit(x_train)

states_train = np.zeros((len(song_id_train),16))
for i, song_id in enumerate(song_id_train):
    song_feats = x_train.loc[[j for j in train_idx if j.split('_')[0] == str(song_id)]]
    song_states = gmm_feat.predict(song_feats)
    states_train[i] = song_states

states_test = np.zeros((len(song_id_test),16))
for i, song_id in enumerate(song_id_test):
    song_feats = x_test.loc[[j for j in test_idx if j.split('_')[0] == str(song_id)]]
    song_states = gmm_feat.predict(song_feats)
    states_test[i] = song_states


##############
## TRAINING ##
##############
#Creating k-folds
folds = 8
kf = model_selection.KFold(n_splits=folds, shuffle=True, random_state=random_state)

#training random forest
budget = 30
min_max_depth = 5
max_max_depth = 20
min_n_estimators = 50
max_n_estimators = 250
step_n_estimators = 50
min_min_samples_leaf = 0.1
max_min_samples_leaf = 0.5
min_min_impurity_decrease = 0.1
max_min_impurity_decrease = 0.5
min_ccp_alpha = 1e-5
max_ccp_alpha = 1e-2

stat_model = pickle.load(open('static_model.pkl', 'rb'))


class random_forest_objective(object):
    def __init__(self, set_train):
        self.set_train = set_train

    def __call__(self, trial):
        n_estimators = trial.suggest_categorical('n_estimators', [i for i in range(min_n_estimators, max_n_estimators+1, step_n_estimators)])
        #criterion = trial.suggest_categorical('criterion', ['mse','mae'])
        max_depth = trial.suggest_int('max_depth', min_max_depth, max_max_depth)
        #min_samples_split = trial.suggest_loguniform('min_samples_split', 0+sys.float_info.min, 0.2)
        min_samples_leaf = trial.suggest_loguniform('min_samples_leaf', min_min_samples_leaf, max_min_samples_leaf)
        #min_weight_fraction_leaf = trial.suggest_loguniform('min_weight_fraction_leaf', 0+sys.float_info.min, 1) 
        #max_features = trial.suggest_uniform('max_features', 0+sys.float_info.min, 1)
        min_impurity_decrease = trial.suggest_loguniform('min_impurity_decrease', min_min_impurity_decrease, max_min_impurity_decrease) 
        ccp_alpha = trial.suggest_loguniform('ccp_alpha', min_ccp_alpha, max_ccp_alpha)

        clf = ensemble.RandomForestRegressor(
            random_state=random_state,
            n_estimators=n_estimators,
            #criterion=criterion,
            max_depth=max_depth,
            #min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            #min_weight_fraction_leaf=min_weight_fraction_leaf,
            #min_impurity_decrease=min_impurity_decrease,
            #max_features=max_features,
            ccp_alpha=ccp_alpha
            )
        
        #we use 3 scores and optimize its mean
        scoring = ['neg_mean_squared_error']
        scores = model_selection.cross_validate(
            clf, states_train, set_train, 
            scoring=scoring, cv = kf)
        
        del scores['fit_time']
        del scores['score_time']

        l = np.asarray([s for k, s in scores.items()])

        #print(l.mean(), l.std())

        return l.mean()


for set_name, set_train, set_test in train_sets:
    rf_optuna = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    rf_optuna.optimize(random_forest_objective(set_train), n_trials=budget)
    models[set_name]['states_gmm']['best_params'] = rf_optuna.best_params

    best = ensemble.RandomForestRegressor(
        random_state=random_state,
        #criterion=models[set_name]['states_gmm']['best_params']['criterion'],
        max_depth=models[set_name]['states_gmm']['best_params']['max_depth'],
        #max_features=models[set_name]['states_gmm']['best_params']['max_features'],
        #min_impurity_decrease=models[set_name]['states_gmm']['best_params']['min_impurity_decrease'],
        min_samples_leaf=models[set_name]['states_gmm']['best_params']['min_samples_leaf'],
        #min_samples_split=models[set_name]['states_gmm']['best_params']['min_samples_split'],
        #min_weight_fraction_leaf=models[set_name][['states_gmm']'best_params']['min_weight_fraction_leaf'],
        n_estimators=models[set_name]['states_gmm']['best_params']['n_estimators'],
        ccp_alpha=models[set_name]['states_gmm']['best_params']['ccp_alpha']
    )
    best = ensemble.RandomForestRegressor()
    best = best.fit(states_train, set_train)
    #set_pred = 0.3*best.predict(states_test)+0.7*stat_model.predict(x_test_stat)
    set_pred = best.predict(states_test)
    models[set_name]['states_gmm']['adjusted_error'] = math.sqrt(metrics.mean_squared_error(set_test, set_pred))/ran(set_test)


#############
## LOADING ##
#############
'''
for set_name, set_train, set_test in train_sets:
    best = ensemble.RandomForestRegressor(
        random_state=random_state,
        #criterion=models[set_name]['static_rf']['best_params']['criterion'],
        max_depth=models[set_name]['static_rf']['best_params']['max_depth'],
        #max_features=models[set_name]['static_rf']['best_params']['max_features'],
        #min_impurity_decrease=models[set_name]['static_rf']['best_params']['min_impurity_decrease'],
        min_samples_leaf=models[set_name]['static_rf']['best_params']['min_samples_leaf'],
        #min_samples_split=models[set_name]['static_rf']['best_params']['min_samples_split'],
        #min_weight_fraction_leaf=models[set_name][['static_rf']'best_params']['min_weight_fraction_leaf'],
        n_estimators=models[set_name]['static_rf']['best_params']['n_estimators'],
        ccp_alpha=models[set_name]['static_rf']['best_params']['ccp_alpha']
    )
    best = ensemble.RandomForestRegressor()
    best = best.fit(x_train, set_train)
    set_pred = best.predict(x_test)
    models[set_name]['static_rf']['adjusted_error'] = math.sqrt(metrics.mean_squared_error(set_test, set_pred))/ran(set_test)'''






with open('models', 'w') as f:
    json.dump(models, f, indent = 4)