import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math

cont = pd.read_parquet('../datasets/emomusic/annotations/cont_selected_features.pqt')
stat = pd.read_parquet('../datasets/emomusic/annotations/static_selected_features.pqt')


inds = stat.index 

def normalizer(x):
	return 2*((x-1)/8)-1

stat['valence_mean'] = stat['valence_mean'].apply(lambda x: normalizer(x))
stat['arousal_mean'] = stat['arousal_mean'].apply(lambda x: normalizer(x))

idx = 2
ind = [i for i in cont.index if str(i).split('_')[0] == str(idx)]
df = cont.loc[ind]

'''mean_diff = []
last_diff = []
for idx in inds:
	ind = [i for i in cont.index if str(i).split('_')[0] == str(idx)]
	df = cont.loc[ind].reset_index()
	
	mean = df['arousal_mean'].mean()
	mean_diff.append(stat.loc[idx]['arousal_mean'] - mean)
	last_diff.append(stat.loc[idx]['arousal_mean'] - df.iloc[-1]['arousal_mean'])

plt.hist(mean_diff)
plt.hist(last_diff)
plt.show()'''

mean_diff = []
last_diff = []
for idx in inds:
	ind = [i for i in cont.index if str(i).split('_')[0] == str(idx)]
	df = cont.loc[ind].reset_index()
	
	mean = df['valence_mean'].mean()
	mean_diff.append(stat.loc[idx]['valence_mean'] - mean)
	last_diff.append(stat.loc[idx]['valence_mean'] - df.iloc[-1]['valence_mean'])

plt.hist(mean_diff)
plt.hist(last_diff)
plt.show()