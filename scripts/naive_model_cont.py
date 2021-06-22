import os
import numpy as np              
import pandas as pd
import matplotlib.pyplot as plt 

import sys
import time
import math
import json

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
clips_path = '../emomusic/clips/'
random_state = 242

#Getting train and test indices
df = pd.read_csv(ann_path+'songs_info.csv')
df = df[['song_id', 'Mediaeval 2013 set']]


t = list(df[df['Mediaeval 2013 set'] == 'development']['song_id'])
train_idx = []
for song_id in t:
    for sam in range(5000, 45001, 2500):
        train_idx.append(str(song_id)+'_'+str(sam))
t = list(df[df['Mediaeval 2013 set'] == 'evaluation']['song_id'])
test_idx = []
for song_id in t:
    for sam in range(5000, 45001, 2500):
        test_idx.append(str(song_id)+'_'+str(sam))


#Loading data for trainning and testing
stat = pd.read_parquet(ann_path+'cont_selected_features_5+25.pqt')
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
va_mean_test = ar_mean.loc[test_idx]
va_std_test = ar_std.loc[test_idx]

#Creating k-folds
folds = 8
kf = model_selection.KFold(n_splits=folds, shuffle=True, random_state=random_state)

#predicting with naive classifier
ar_mean_pred = np.full(ar_mean_test.shape, np.mean(ar_mean_train))
ar_std_pred = np.full(ar_std_test.shape, np.mean(ar_std_train))
va_mean_pred = np.full(va_mean_test.shape, np.mean(va_mean_train))
va_std_pred = np.full(va_std_test.shape, np.mean(va_std_train))

#Getting adjusted mse
def ran(l):
    return np.ptp(l)
ar_mean_ad_mse = math.sqrt(metrics.mean_squared_error(ar_mean_test, ar_mean_pred))/ran(ar_mean_test)
ar_std_ad_mse = math.sqrt(metrics.mean_squared_error(ar_std_test, ar_std_pred))/ran(ar_std_test)
va_mean_ad_mse = math.sqrt(metrics.mean_squared_error(va_mean_test, va_mean_pred))/ran(va_mean_test)
va_std_ad_mse = math.sqrt(metrics.mean_squared_error(va_std_test, va_std_pred))/ran(va_std_test)


with open('models', 'r') as f:
    models = json.load(f)

models['arousal_mean']['transitions_hmm']['adjusted_error'] = ar_mean_ad_mse
models['arousal_std']['transitions_hmm']['adjusted_error'] = ar_std_ad_mse
models['valence_mean']['transitions_hmm']['adjusted_error'] = va_mean_ad_mse
models['valence_std']['transitions_hmm']['adjusted_error'] = va_std_ad_mse


with open('models', 'w') as f:
    json.dump(models, f, indent = 4)