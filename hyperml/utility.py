from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve
import pandas as pd
import json
from sklearn.utils import shuffle


def load_ML_model_cfg(args, task_type):
    if task_type != "classification" and task_type != "regression":
        raise ValueError(f'Unknown task type: {task_type}')

    if(args.model == "KNN"):
        if(task_type == "classification"):
            link = './cfg/ml_model/KNeighborsClassifier_config.json'
        else:
            link = './cfg/ml_model/KNeighborsRegressor_config.json'

    elif(args.model == "MLP"):
        if(task_type == "classification"):
            link = './cfg/ml_model/MLPClassifier_config.json'
        else:
            link = './cfg/ml_model/MLPRegressor_config.json'

    elif(args.model == "SVM"):
        if(task_type == "classification"):
            link = './cfg/ml_model/SVC_config.json'
        else:
            link = './cfg/ml_model/SVR_config.json'

    elif(args.model == "RF"):
        if(task_type == "classification"):
            link = './cfg/ml_model/RandomForestClassifier_config.json'
        else:
            link = './cfg/ml_model/RandomForestRegressor_config.json'

    elif(args.model == "ADA"):
        if(task_type == "classification"):
            link = './cfg/ml_model/AdaBoostClassifier_config.json'
        else:
            link = './cfg/ml_model/AdaBoostRegressor_config.json'

    elif(args.model == "XGBoost"):
        if(task_type == "classification"):
            link = './cfg/ml_model/XGBClassifier_config.json'
        else:
            link = './cfg/ml_model/XGBRegressor_config.json'

    with open(link) as f:
        model_cfg = json.load(f)

    return model_cfg

def load_data(cfg):
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