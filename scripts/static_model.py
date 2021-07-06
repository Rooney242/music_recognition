import os
import numpy as np              
import pandas as pd
import matplotlib.pyplot as plt 

import sys
import time
import math
import json
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, model_selection, metrics, neighbors, ensemble, feature_selection
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import optuna
import optuna.visualization as ov

ann_path = '../emomusic/annotations/'
mod_path = '../models/'
clips_path = '../emomusic/clips/'
random_state = 333

#configuration
conf = json.load(open('models_conf', 'r'))


#Getting train and test indices
df = pd.read_csv(ann_path+'songs_info.csv')
df = df[['song_id', 'Mediaeval 2013 set']]

train_idx = list(df[df['Mediaeval 2013 set'] == 'development']['song_id'])
test_idx = list(df[df['Mediaeval 2013 set'] == 'evaluation']['song_id'])


#Loading data for trainning and testing
stat = pd.read_parquet(ann_path+'static_selected_features.pqt')
x = stat.drop(['arousal_mean', 'arousal_std', 'valence_mean', 'valence_std'], axis=1)
ar_mean = stat['arousal_mean']
ar_std = stat['arousal_std']
va_mean = stat['valence_mean']
va_std = stat['valence_std']

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

#train_sets = [('arousal_mean', ar_mean_train, ar_mean_test), ('arousal_std', ar_std_train, ar_std_test), ('valence_mean', va_mean_train, va_mean_test), ('valence_std', va_std_train, va_std_test)]
train_sets = [('arousal_mean', ar_mean_train, ar_mean_test), ('valence_mean', va_mean_train, va_mean_test)]


#models information
with open('models', 'r') as f:
    models = json.load(f)

##############
## TRAINING ##
##############
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
        scoring = ['r2']
        scores = model_selection.cross_validate(
            clf, x_train, set_train, 
            scoring=scoring, cv = kf)
        
        del scores['fit_time']
        del scores['score_time']

        l = np.asarray([s for k, s in scores.items()])

        if conf['norm_std']:
            return l.mean()/l.std()
        else:
            return l.mean()


for set_name, set_train, set_test in train_sets:
    rf_optuna = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    rf_optuna.optimize(random_forest_objective(set_train), n_trials=budget)
    models[set_name]['static']['best_params'] = rf_optuna.best_params

    best = ensemble.RandomForestRegressor(
        random_state=random_state,
        #criterion=models[set_name]['static']['best_params']['criterion'],
        max_depth=models[set_name]['static']['best_params']['max_depth'],
        #max_features=models[set_name]['static']['best_params']['max_features'],
        #min_impurity_decrease=models[set_name]['static']['best_params']['min_impurity_decrease'],
        min_samples_leaf=models[set_name]['static']['best_params']['min_samples_leaf'],
        #min_samples_split=models[set_name]['static']['best_params']['min_samples_split'],
        #min_weight_fraction_leaf=models[set_name][['static']'best_params']['min_weight_fraction_leaf'],
        n_estimators=models[set_name]['static']['best_params']['n_estimators'],
        ccp_alpha=models[set_name]['static']['best_params']['ccp_alpha']
    )
    best = ensemble.RandomForestRegressor()
    best = best.fit(x_train, set_train)
    set_pred = best.predict(x_test)
    models[set_name]['static']['r2'] = metrics.r2_score(set_test, set_pred)
    pickle.dump(best, open(mod_path+'static_'+set_name+'.pkl', 'wb'))


#############
## LOADING ##
#############

'''for set_name, set_train, set_test in train_sets:
    best = ensemble.RandomForestRegressor(
        random_state=random_state,
        #criterion=models[set_name]['static']['best_params']['criterion'],
        max_depth=models[set_name]['static']['best_params']['max_depth'],
        #max_features=models[set_name]['static']['best_params']['max_features'],
        #min_impurity_decrease=models[set_name]['static']['best_params']['min_impurity_decrease'],
        min_samples_leaf=models[set_name]['static']['best_params']['min_samples_leaf'],
        #min_samples_split=models[set_name]['static']['best_params']['min_samples_split'],
        #min_weight_fraction_leaf=models[set_name][['static']'best_params']['min_weight_fraction_leaf'],
        n_estimators=models[set_name]['static']['best_params']['n_estimators'],
        ccp_alpha=models[set_name]['static']['best_params']['ccp_alpha']
    )
    best = ensemble.RandomForestRegressor()
    best = best.fit(x_train, set_train)
    set_pred = best.predict(x_test)
    models[set_name]['static']['adjusted_error'] = metrics.r2_score(set_test, set_pred)'''


with open('models', 'w') as f:
    json.dump(models, f, indent = 4)