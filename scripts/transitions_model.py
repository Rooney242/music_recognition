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
random_state = 222

#configuration
conf = json.load(open('models_conf', 'r'))
mod_name = 'transitions_'+str(int(int(conf['window_size_ms'])/1000))
mod_name += '_ov' if conf['window_shift'] else ''

range_start = conf['window_size_ms']
range_end = 45001
range_step = int(range_start/2) if conf['window_shift'] else range_start

#Getting train and test indices
df = pd.read_csv(ann_path+'songs_info.csv')
df = df[['song_id', 'Mediaeval 2013 set']]

song_id_train = list(df[df['Mediaeval 2013 set'] == 'development']['song_id'])
train_idx = []
for song_id in song_id_train:
    for sam in range(range_start, range_end, range_step):
        train_idx.append(str(song_id)+'_'+str(sam))

song_id_test = list(df[df['Mediaeval 2013 set'] == 'evaluation']['song_id'])
test_idx = []
for song_id in song_id_test:
    for sam in range(range_start, range_end, range_step):
        test_idx.append(str(song_id)+'_'+str(sam))

#Loading data for trainning and testing
cont = pd.read_parquet(ann_path+'cont_selected_features_'+str(range_start)+'.pqt')

x = cont
x_train = x.loc[train_idx]
x_test = x.loc[test_idx]

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

#train_sets = [('arousal_mean', ar_mean_train_stat, ar_mean_test_stat), ('arousal_std', ar_std_train_stat, ar_std_test_stat), ('valence_mean', va_mean_train_stat, va_mean_test_stat), ('valence_std', va_std_train_stat, va_std_test_stat)]
train_sets = [('arousal_mean', ar_mean_train_stat, ar_mean_test_stat), ('valence_mean', va_mean_train_stat, va_mean_test_stat)]

#models information
with open('models', 'r') as f:
    models = json.load(f)

#################
## TRANSITIONS ##
#################
np.set_printoptions(threshold=sys.maxsize)
stat_model_arousal = pickle.load(open(mod_path+'static_arousal_mean.pkl', 'rb'))
stat_model_valence = pickle.load(open(mod_path+'static_valence_mean.pkl', 'rb'))

mus_ele = list(dict.fromkeys([i.split('.')[0] for i in x.columns]))
k_list = range(3,12)
mus_k = []
gmm_feat = {}

for m in mus_ele:
    x_me = x.filter(regex='^'+m, axis=1)

    bic_list = []
    for k in k_list:
        gmm = mixture.GaussianMixture(n_components=k,covariance_type='diag').fit(x_me)
        bic_list.append(gmm.bic(x_me))
    min_k = k_list[bic_list.index(min(bic_list))]
    mus_k.append(min_k)
    gmm_feat[m] = mixture.GaussianMixture(n_components=min_k,covariance_type='diag').fit(x_me)

num_states = len(range(range_start, range_end, range_step))
trans_train_arousal = np.zeros((len(song_id_train), num_states*sum(mus_k)+x_train_stat.shape[1]))
trans_train_valence = np.zeros((len(song_id_train), num_states*sum(mus_k)+x_train_stat.shape[1]))
for i, song_id in enumerate(song_id_train):
    cont = 0
    for j, m in enumerate(mus_ele):
        song_feats = x_train.loc[[j for j in train_idx if j.split('_')[0] == str(song_id)]]
        song_feats = song_feats.filter(regex='^'+m, axis=1)
        song_states = gmm_feat[m].predict(song_feats)
        num_states = len(song_states)

        for st in range(num_states):
            for tr in range(mus_k[j]):
                trans_train_arousal[i][cont+st*mus_k[j]+tr] = trans_train_valence[i][cont+st*mus_k[j]+tr] = 1 if song_states[st] == tr else 0
        cont += mus_k[j]*num_states
    
    trans_train_arousal[i][num_states*sum(mus_k):] = x_train_stat.loc[song_id].to_numpy().reshape(1,-1)
    trans_train_valence[i][num_states*sum(mus_k):] = x_train_stat.loc[song_id].to_numpy().reshape(1,-1)


num_states = len(range(range_start, range_end, range_step))
trans_test_arousal = np.zeros((len(song_id_test), num_states*sum(mus_k)++x_test_stat.shape[1]))
trans_test_valence = np.zeros((len(song_id_test), num_states*sum(mus_k)++x_test_stat.shape[1]))
for i, song_id in enumerate(song_id_test):
    cont = 0
    for j, m in enumerate(mus_ele):
        song_feats = x_test.loc[[j for j in test_idx if j.split('_')[0] == str(song_id)]]
        song_feats = song_feats.filter(regex='^'+m, axis=1)
        song_states = gmm_feat[m].predict(song_feats)
        num_states = len(song_states)

        for st in range(num_states):
            for tr in range(mus_k[j]):
                trans_test_arousal[i][cont+st*mus_k[j]+tr] = trans_test_valence[i][cont+st*mus_k[j]+tr] = 1 if song_states[st] == tr else 0
        cont += mus_k[j]*num_states
    
    trans_test_arousal[i][num_states*sum(mus_k):] = x_test_stat.loc[song_id].to_numpy().reshape(1,-1)
    trans_test_valence[i][num_states*sum(mus_k):] = x_test_stat.loc[song_id].to_numpy().reshape(1,-1)



##############
## TRAINING ##
##############
#train_sets = [('arousal_mean', ar_mean_train_stat, ar_mean_test_stat), ('arousal_std', ar_std_train_stat, ar_std_test_stat), ('valence_mean', va_mean_train_stat, va_mean_test_stat), ('valence_std', va_std_train_stat, va_std_test_stat)]
train_sets = [
    ('arousal_mean', trans_train_arousal, trans_test_arousal, ar_mean_train_stat, ar_mean_test_stat), 
    ('valence_mean', trans_train_valence, trans_test_valence, va_mean_train_stat, va_mean_test_stat)
]


#Creating k-folds
folds = 8
kf = model_selection.KFold(n_splits=folds, shuffle=True, random_state=random_state)

#training random forest
budget = conf['budget']
min_max_depth = conf['min_max_depth']
max_max_depth = conf['max_max_depth']
min_n_estimators = conf['min_n_estimators']
max_n_estimators = conf['max_n_estimators']
step_n_estimators = conf['step_n_estimators']
min_min_samples_leaf = conf['min_min_samples_leaf']
max_min_samples_leaf = conf['max_min_samples_leaf']
min_min_impurity_decrease = conf['min_min_impurity_decrease']
max_min_impurity_decrease = conf['max_min_impurity_decrease']
min_ccp_alpha = conf['min_ccp_alpha']
max_ccp_alpha = conf['max_ccp_alpha']


class random_forest_objective(object):
    def __init__(self, feat_train, set_train):
        self.feat_train = feat_train
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
            min_impurity_decrease=min_impurity_decrease,
            #max_features=max_features,
            ccp_alpha=ccp_alpha
            )
        
        #we use 3 scores and optimize its mean
        scoring = ['r2']
        scores = model_selection.cross_validate(
            clf, feat_train, set_train, 
            scoring=scoring, cv = kf)
        
        del scores['fit_time']
        del scores['score_time']

        l = np.asarray([s for k, s in scores.items()])

        if conf['norm_std']:
            return l.mean()/l.std()
        else:
            return l.mean()


for set_name, feat_train, feat_test, set_train, set_test in train_sets:
    rf_optuna = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    rf_optuna.optimize(random_forest_objective(feat_train, set_train), n_trials=budget)
    models[set_name][mod_name]['best_params'] = rf_optuna.best_params

    best = ensemble.RandomForestRegressor(
        random_state=random_state,
        #criterion=models[set_name][mod_name]['best_params']['criterion'],
        max_depth=models[set_name][mod_name]['best_params']['max_depth'],
        #max_features=models[set_name][mod_name]['best_params']['max_features'],
        min_impurity_decrease=models[set_name][mod_name]['best_params']['min_impurity_decrease'],
        min_samples_leaf=models[set_name][mod_name]['best_params']['min_samples_leaf'],
        #min_samples_split=models[set_name][mod_name]['best_params']['min_samples_split'],
        #min_weight_fraction_leaf=models[set_name][[mod_name]'best_params']['min_weight_fraction_leaf'],
        n_estimators=models[set_name][mod_name]['best_params']['n_estimators'],
        ccp_alpha=models[set_name][mod_name]['best_params']['ccp_alpha']
    )
    best = ensemble.RandomForestRegressor()
    best = best.fit(feat_train, set_train)
    set_pred = best.predict(feat_test)
    models[set_name][mod_name]['r2'] = metrics.r2_score(set_test, set_pred)
    #pickle.dump(best, open(mod_path+mod_name+'_'+set_name+'.pkl', 'wb'))


#############
## LOADING ##
#############
'''
for set_name, set_train, set_test in train_sets:
    best = ensemble.RandomForestRegressor(
        random_state=random_state,
        #criterion=models[set_name][mod_name]['best_params']['criterion'],
        max_depth=models[set_name][mod_name]['best_params']['max_depth'],
        #max_features=models[set_name][mod_name]['best_params']['max_features'],
        #min_impurity_decrease=models[set_name][mod_name]['best_params']['min_impurity_decrease'],
        min_samples_leaf=models[set_name][mod_name]['best_params']['min_samples_leaf'],
        #min_samples_split=models[set_name][mod_name]['best_params']['min_samples_split'],
        #min_weight_fraction_leaf=models[set_name][[mod_name]'best_params']['min_weight_fraction_leaf'],
        n_estimators=models[set_name][mod_name]['best_params']['n_estimators'],
        ccp_alpha=models[set_name][mod_name]['best_params']['ccp_alpha']
    )
    best = ensemble.RandomForestRegressor()
    best = best.fit(x_train, set_train)
    set_pred = best.predict(x_test)
    models[set_name][mod_name]['r2'] = math.sqrt(metrics.mean_squared_error(set_test, set_pred))/ran(set_test)'''

with open('models', 'w') as f:
    json.dump(models, f, indent = 4)