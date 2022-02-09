from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve
import pandas as pd
import plotly.express as px

def load_KNN_cfg(model_cfg):
        KNN_cfg = {
            "max_value_neighbors" : model_cfg['max_value_neighbors'],
            "min_value_neighbors" : model_cfg['min_value_neighbors'],
            "max_value_leaf_size" : model_cfg['max_value_leaf_size'],
            "min_value_leaf_size" : model_cfg['min_value_leaf_size'],
            "max_value_weights" : model_cfg['max_value_weights'],
            "min_value_weights" : model_cfg['min_value_weights'],
            "max_value_algorithm" : model_cfg['max_value_algorithm'],
            "min_value_algorithm" : model_cfg['min_value_algorithm'],
            "max_value_metric" : model_cfg['max_value_metric'],
            "min_value_metric" : model_cfg['min_value_metric'],
            "int_parameter": model_cfg['int_parameter']
        }
        return KNN_cfg


def load_MLP_cfg(model_cfg):
        MLP_cfg = {
            "max_value_hidden_layer": model_cfg['max_value_hidden_layer'],
            "min_value_hidden_layer": model_cfg['min_value_hidden_layer'],
            "max_value_alpha": model_cfg['max_value_alpha'],
            "min_value_alpha": model_cfg['min_value_alpha'],
            "max_value_learning_rate_init": model_cfg['max_value_learning_rate_init'],
            "min_value_learning_rate_init": model_cfg['min_value_learning_rate_init'],
            "max_value_max_iter": model_cfg['max_value_max_iter'],
            "min_value_max_iter": model_cfg['min_value_max_iter'],
            "max_value_tol": model_cfg['max_value_tol'],
            "min_value_tol": model_cfg['min_value_tol'],
            "max_value_beta": model_cfg['max_value_beta'],
            "min_value_beta": model_cfg['min_value_beta'],
            "max_value_n_iter_no_change": model_cfg['max_value_n_iter_no_change'],
            "min_value_n_iter_no_change": model_cfg['min_value_n_iter_no_change'],
            "max_value_activation": model_cfg['max_value_activation'],
            "min_value_activation": model_cfg['min_value_activation'],
            "max_value_solver": model_cfg['max_value_solver'],
            "min_value_solver": model_cfg['min_value_solver'],
            "max_value_learning_rate": model_cfg['max_value_learning_rate'],
            "min_value_learning_rate": model_cfg['min_value_learning_rate'],
            "int_parameter": model_cfg['int_parameter']
        }
        return MLP_cfg


def load_SVM_cfg(model_cfg):
        SVM_cfg = {
            "max_value_c": model_cfg['max_value_c'],
            "min_value_c": model_cfg['min_value_c'],
            "max_value_tol": model_cfg['max_value_tol'],
            "min_value_tol": model_cfg['min_value_tol'],
            "max_value_max_iter": model_cfg['max_value_max_iter'],
            "min_value_max_iter": model_cfg['min_value_max_iter'],
            "max_value_gamma": model_cfg['max_value_gamma'],
            "min_value_gamma": model_cfg['min_value_gamma'],
            "max_value_kernal": model_cfg['max_value_kernal'],
            "min_value_kernal": model_cfg['min_value_kernal'],
            "int_parameter": model_cfg['int_parameter']
        }
        return SVM_cfg

def load_data(data,cfg=None):
        if(data == "cancer"):
                data_ovarian = pd.read_csv(
                "C:/Users/Yang/Desktop/data/ovarian cancer.csv")
                data_breast = pd.read_csv(
                "C:/Users/Yang/Desktop/data/Breast cancer.csv")
                data_cervical = pd.read_csv(
                "C:/Users/Yang/Desktop/data/Cervical cancer.csv")
                dataset = [data_ovarian, data_breast, data_cervical]

        elif(data == "machine"):
                data_machine = pd.read_csv("C:/Users/Yang/Desktop/data/cooler-Valve-Pump-Accumulator(random).csv")
                dataset = [data_machine]
        elif(data == "custom"):
                data = pd.read_csv(cfg['data_path'])
                dataset = [data]
        return dataset


def plot_boxplot(data, boxpath):
    fig = px.box(data, y="test",title=boxpath[:-4])
    fig.write_image(boxpath)


def specificity_func(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    speci = 1 - fpr
    return speci

def customize_speci():
    specificity = make_scorer(specificity_func)
    return specificity
