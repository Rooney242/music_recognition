import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

graph_path = '../graphs/'


with open('models', 'r') as f:
    models = json.load(f)

info = pd.DataFrame(columns=models.keys())

for metric in models.keys():
	for model in models[metric].keys():
		err = round(models[metric][model]['r2'], 3) if 'r2' in models[metric][model].keys() else 0
		info.loc[model, metric] = err

'''labels_st = ('Static', 'States (5s)', 'States (10s)', 'States (15s)')
labels_tr = ('Static', 'Transitions (5s)', 'Transitions (10s)', 'Transitions (15s)')
bar_width = 0.25

states = info.loc[['static', 'states_5', 'states_10', 'states_15']]
rs = np.arange(states.shape[0])
states_ov = info.loc[['static', 'states_5_ov', 'states_10_ov', 'states_15_ov']]
rso = [i + bar_width for i in rs]

plt.bar(rs, states['arousal_mean'], color=('red', 'blue', 'blue', 'blue'), width=bar_width, label='no overlap')
plt.bar(rso, states_ov['arousal_mean'], color=('red', 'green', 'green', 'green'), width=bar_width, label='overlap')
plt.xticks([r + bar_width/2 for r in range(states.shape[0])], labels_st)
plt.title('Performance of states models for predicting arousal')
plt.xlabel('Model')
plt.ylabel('R square')
ax = plt.gca()
leg = ax.legend()
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('green')
plt.savefig(graph_path+'states_arousal.png')
plt.clf()

plt.bar(rs, states['valence_mean'], color=('red', 'blue', 'blue', 'blue'), width=bar_width, label='no overlap')
plt.bar(rso, states_ov['valence_mean'], color=('red', 'green', 'green', 'green'), width=bar_width, label='overlap')
plt.xticks([r + bar_width/2 for r in range(states.shape[0])], labels_st)
plt.title('Performance of states models for predicting valence')
plt.xlabel('Model')
plt.ylabel('R square')
ax = plt.gca()
leg = ax.legend(loc='center left')
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('green')
plt.savefig(graph_path+'states_valence.png')
plt.clf()


trans = info.loc[['static', 'transitions_5', 'transitions_10', 'transitions_15']]
rs = np.arange(trans.shape[0])
states_ov = info.loc[['static', 'transitions_5_ov', 'transitions_10_ov', 'transitions_15_ov']]
rso = [i + bar_width for i in rs]

plt.bar(rs, trans['arousal_mean'], color=('red', 'blue', 'blue', 'blue'), width=bar_width, label='no overlap')
plt.bar(rso, states_ov['arousal_mean'], color=('red', 'green', 'green', 'green'), width=bar_width, label='overlap')
plt.xticks([r + bar_width/2 for r in range(trans.shape[0])], labels_tr)
plt.title('Performance of transitions models for predicting arousal')
plt.xlabel('Model')
plt.ylabel('R square')
ax = plt.gca()
leg = ax.legend(loc='upper left')
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('green')
plt.savefig(graph_path+'transitions_arousal.png')
plt.clf()

plt.bar(rs, trans['valence_mean'], color=('red', 'blue', 'blue', 'blue'), width=bar_width, label='no overlap')
plt.bar(rso, states_ov['valence_mean'], color=('red', 'green', 'green', 'green'), width=bar_width, label='overlap')
plt.xticks([r + bar_width/2 for r in range(trans.shape[0])], labels_tr)
plt.title('Performance of transitions models for predicting valence')
plt.xlabel('Model')
plt.ylabel('R square')
ax = plt.gca()
leg = ax.legend(loc='center left')
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('green')
plt.savefig(graph_path+'transitions_valence.png')
plt.clf()'''

labels_cnn = ('Static model', 'Best synthetic model', 'Convolutional model')
bar_width = 0.25

cnn_arousal = info.loc[['static', 'transitions_5_ov', 'cnn']]
rs = np.arange(cnn_arousal.shape[0])
plt.bar(rs, cnn_arousal['arousal_mean'], color=('red', 'blue', 'green'), width=bar_width)
#plt.hlines(0.54, xmin=-1, xmax=3, label='baseline')
plt.xticks([r for r in range(cnn_arousal.shape[0])], labels_cnn)
plt.title('Performance of convolutional model for predicting arousal')
plt.ylabel('R square')
plt.savefig(graph_path+'cnn_arousal.png')
plt.clf()


cnn_valence = info.loc[['static', 'transitions_15_ov', 'cnn']]
rs = np.arange(cnn_valence.shape[0])
plt.bar(rs, cnn_valence['valence_mean'], color=('red', 'blue', 'green'), width=bar_width)
#plt.hlines(0.07, xmin=-1, xmax=3, label='baseline')
plt.xticks([r for r in range(cnn_valence.shape[0])], labels_cnn)
plt.title('Performance of convolutional model for predicting valence')
plt.ylabel('R square')
plt.savefig(graph_path+'cnn_valence.png')
plt.clf()