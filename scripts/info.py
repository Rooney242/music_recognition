import pandas as pd
import json


with open('models', 'r') as f:
    models = json.load(f)


info = pd.DataFrame(columns=models.keys())

for metric in models.keys():
	for model in models[metric].keys():
		err = round(models[metric][model]['r2'], 3) if 'r2' in models[metric][model].keys() else 0
		info.loc[model, metric] = err


#print(info.sort_values(by=['valence_mean']))
print(info)