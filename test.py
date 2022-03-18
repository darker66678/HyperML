import json

with open('./cfg/ml_model/KNN_config.json') as f:
    model_cfg_json = json.load(f)
model_cfg = model_cfg_json
a = []
for index, i in enumerate(model_cfg.keys()):
    if model_cfg[i]['int']:
        a.append(index)
print(a)
