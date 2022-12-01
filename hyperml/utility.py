from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve
import pandas as pd
import json
from sklearn.utils import shuffle


def load_ML_model_cfg(args):
    if(args.model == "KNN"):
        link = './cfg/ml_model/KNN_config.json'
    elif(args.model == "MLP"):
        link = './cfg/ml_model/MLP_config.json'
    elif(args.model == "SVM"):
        link = './cfg/ml_model/SVM_config.json'
    elif(args.model == "RF"):
        link = './cfg/ml_model/RF_config.json'
    elif(args.model == "ADA"):
        link = './cfg/ml_model/ADA_config.json'
    elif(args.model == "XGBoost"):
        link = './cfg/ml_model/XGBoost_config.json'
    with open(link) as f:
        model_cfg = json.load(f)

    return model_cfg

def load_data(data, cfg):
    data = pd.read_csv(cfg['data_path'])
    shuffle_data = shuffle(data, random_state=1)
    dataset = [shuffle_data]
    task_type = cfg['type']
    y = shuffle_data[cfg['target']].reset_index(drop=True)
    X = shuffle_data.drop([cfg['target']], axis=1).reset_index(drop=True)
    target = cfg['target']
    data_path = cfg['data_path']
    file = cfg['data_name']
    return dataset, task_type, y, X, file, target, data_path

def specificity_func(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    speci = 1 - fpr
    return speci

def customize_speci():
    specificity = make_scorer(specificity_func)
    return specificity