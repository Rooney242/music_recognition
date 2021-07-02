import pandas as pd
import json


with open('models', 'r') as f:
    models = json.load(f)


info = pd.DataFrame(columns=models.keys())

for metric in models.keys():
	for model in models[metric].keys():
		err = models[metric][model]['adjusted_error'] if 'adjusted_error' in models[metric][model].keys() else 0
		info.loc[model, metric] = err


info = info[['arousal_mean', 'valence_mean']]
print(info)