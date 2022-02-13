from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve
import pandas as pd
import plotly.express as px
import json
import datetime
import logging
from PSO import PSO
from VOA import VOA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_KNN_cfg(model_cfg):
    KNN_cfg = {
        "max_value_neighbors": model_cfg['param_setting']['max_value_neighbors'],
        "min_value_neighbors": model_cfg['param_setting']['min_value_neighbors'],
        "max_value_leaf_size": model_cfg['param_setting']['max_value_leaf_size'],
        "min_value_leaf_size": model_cfg['param_setting']['min_value_leaf_size'],
        "max_value_weights": model_cfg['param_setting']['max_value_weights'],
        "min_value_weights": model_cfg['param_setting']['min_value_weights'],
        "max_value_algorithm": model_cfg['param_setting']['max_value_algorithm'],
        "min_value_algorithm": model_cfg['param_setting']['min_value_algorithm'],
        "max_value_metric": model_cfg['param_setting']['max_value_metric'],
        "min_value_metric": model_cfg['param_setting']['min_value_metric'],
        "int_parameter": model_cfg['int_param'],
        "class_parameter": model_cfg['class_param'],
        "param": model_cfg['param']
    }
    return KNN_cfg


def load_MLP_cfg(model_cfg):
    MLP_cfg = {
        "max_value_hidden_layer_1": model_cfg['param_setting']['max_value_hidden_layer_1'],
        "min_value_hidden_layer_1": model_cfg['param_setting']['min_value_hidden_layer_1'],
        "max_value_hidden_layer_2": model_cfg['param_setting']['max_value_hidden_layer_2'],
        "min_value_hidden_layer_2": model_cfg['param_setting']['min_value_hidden_layer_2'],
        "max_value_alpha": model_cfg['param_setting']['max_value_alpha'],
        "min_value_alpha": model_cfg['param_setting']['min_value_alpha'],
        "max_value_learning_rate_init": model_cfg['param_setting']['max_value_learning_rate_init'],
        "min_value_learning_rate_init": model_cfg['param_setting']['min_value_learning_rate_init'],
        "max_value_max_iter": model_cfg['param_setting']['max_value_max_iter'],
        "min_value_max_iter": model_cfg['param_setting']['min_value_max_iter'],
        "max_value_tol": model_cfg['param_setting']['max_value_tol'],
        "min_value_tol": model_cfg['param_setting']['min_value_tol'],
        "max_value_beta_1": model_cfg['param_setting']['max_value_beta_1'],
        "min_value_beta_1": model_cfg['param_setting']['min_value_beta_1'],
        "max_value_beta_2": model_cfg['param_setting']['max_value_beta_2'],
        "min_value_beta_2": model_cfg['param_setting']['min_value_beta_2'],
        "max_value_n_iter_no_change": model_cfg['param_setting']['max_value_n_iter_no_change'],
        "min_value_n_iter_no_change": model_cfg['param_setting']['min_value_n_iter_no_change'],
        "max_value_activation": model_cfg['param_setting']['max_value_activation'],
        "min_value_activation": model_cfg['param_setting']['min_value_activation'],
        "max_value_solver": model_cfg['param_setting']['max_value_solver'],
        "min_value_solver": model_cfg['param_setting']['min_value_solver'],
        "max_value_learning_rate": model_cfg['param_setting']['max_value_learning_rate'],
        "min_value_learning_rate": model_cfg['param_setting']['min_value_learning_rate'],
        "int_parameter": model_cfg['int_param'],
        "class_parameter": model_cfg['class_param'],
        "param": model_cfg['param']
    }
    return MLP_cfg


def load_SVM_cfg(model_cfg):
    SVM_cfg = {
        "max_value_c": model_cfg['param_setting']['max_value_c'],
        "min_value_c": model_cfg['param_setting']['min_value_c'],
        "max_value_tol": model_cfg['param_setting']['max_value_tol'],
        "min_value_tol": model_cfg['param_setting']['min_value_tol'],
        "max_value_max_iter": model_cfg['param_setting']['max_value_max_iter'],
        "min_value_max_iter": model_cfg['param_setting']['min_value_max_iter'],
        "max_value_gamma": model_cfg['param_setting']['max_value_gamma'],
        "min_value_gamma": model_cfg['param_setting']['min_value_gamma'],
        "max_value_kernal": model_cfg['param_setting']['max_value_kernal'],
        "min_value_kernal": model_cfg['param_setting']['min_value_kernal'],
        "int_parameter": model_cfg['int_param'],
        "class_parameter": model_cfg['class_param'],
        "param": model_cfg['param']
    }
    return SVM_cfg


def load_ML_model_cfg(args):
    if(args.model == "KNN"):
        with open('./cfg/ml_model/KNN_config.json') as f:
            model_cfg_json = json.load(f)
        model_cfg = load_KNN_cfg(model_cfg_json)

    elif(args.model == "MLP"):
        with open('./cfg/ml_model/MLP_config.json') as f:
            model_cfg_json = json.load(f)
        model_cfg = load_MLP_cfg(model_cfg_json)

    elif(args.model == "SVM"):
        with open('./cfg/ml_model/SVM_config.json') as f:
            model_cfg_json = json.load(f)
        model_cfg = load_SVM_cfg(model_cfg_json)
    return model_cfg, model_cfg_json


def load_data(data, cfg):
    data = pd.read_csv(cfg['data_path'])
    dataset = [data]
    task_type = cfg['type']
    y = data[cfg['target']]
    X = data.drop([cfg['target']], axis=1)
    target = cfg['target']
    data_path = cfg['data_path']
    file = cfg['data_name']
    return dataset, task_type, y, X, file, target, data_path


def plot_boxplot(data, boxpath):
    fig = px.box(data, y="test", title=boxpath[:-4].split("/")[-1])
    fig.write_image(boxpath)


def specificity_func(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    speci = 1 - fpr
    return speci


def customize_speci():
    specificity = make_scorer(specificity_func)
    return specificity


def hyper(args, model_cfg, X, y, file, folder, task_type):
    if(args.algo == "PSO"):
        with open('./cfg/algo/PSO_config.json') as f:
            algo_cfg = json.load(f)
            print(f'PSO parameters: {algo_cfg}')
            logging.info(f'PSO config: {algo_cfg}')
        particle_num = algo_cfg['particle_num']
        particle_dim = int((len(model_cfg)-3)/2)
        iter_num = algo_cfg['iter_num']
        c1 = algo_cfg['c1']
        c2 = algo_cfg['c2']
        w = algo_cfg['w']

        starttime = datetime.datetime.now()
        pso = PSO(particle_num, particle_dim, iter_num, c1,
                  c2, w, model_cfg, X, y, file, args.model, folder, args.scoring, task_type, algo_MLconfig)
        results, ML_model, best_parameter, class_param = pso.main()
        endtime = datetime.datetime.now()
        logging.info(f"time: {endtime - starttime}")
        logging.info("-----------------------\n\n")
        print("time: ", (endtime - starttime))
        print("---------------------------")
        results.to_csv(
            f'{folder}/{args.algo}({args.model})_{file}.csv')

    elif(args.algo == "VOA"):
        with open('./cfg/algo/VOA_config.json') as f:
            algo_cfg = json.load(f)
            print(f'VOA config: {algo_cfg}')
            logging.info(f'VOA config: {algo_cfg}')

        virus_num = algo_cfg['virus_num']
        virus_dim = int((len(model_cfg)-3)/2)
        s_proportion = algo_cfg['s_proportion']
        strong_growth_rate = algo_cfg['strong_growth_rate']
        common_growth_rate = algo_cfg['common_growth_rate']
        total_virus_limit = algo_cfg['total_virus_limit']
        intensity = algo_cfg['intensity']
        starttime = datetime.datetime.now()
        voa = VOA(virus_num, virus_dim, strong_growth_rate,
                  common_growth_rate, s_proportion, total_virus_limit, intensity, model_cfg, X, y, args.model, args.scoring, task_type, folder, file, algo_MLconfig)
        results, ML_model, best_parameter, class_param = voa.main()
        endtime = datetime.datetime.now()
        logging.info((f"time:{endtime - starttime} "))
        logging.info("-----------------------\n\n")
        print("time: ", (endtime - starttime))
        print("---------------------------")
        results.to_csv(
            f'{folder}/{args.algo}({args.model})_{file}.csv')

    boxpath = f'{folder}/{args.algo}_{args.model}_{file}_box.jpg'
    plot_boxplot(results, boxpath)
    predict_data = model_predict(
        args.model, ML_model, best_parameter, class_param, X, y, task_type, folder)
    predict_data.to_csv(
        f'{folder}/{args.algo}({args.model})_{file}_train_predict.csv')
    print("finished!!!")


def algo_MLconfig(model, task_type, model_cfg):
    if(model == "KNN"):
        neighbors = [model_cfg["max_value_neighbors"],
                     model_cfg["min_value_neighbors"]]
        leaf_size = [model_cfg["max_value_leaf_size"],
                     model_cfg["min_value_leaf_size"]]
        weights = [model_cfg["max_value_weights"],
                   model_cfg["min_value_weights"]]
        algorithm = [model_cfg["max_value_algorithm"],
                     model_cfg["min_value_algorithm"]]
        metric = [model_cfg["max_value_metric"],
                  model_cfg["min_value_metric"]]
        max_min = [neighbors, leaf_size, weights, algorithm, metric]
        if task_type == "classification":
            ML_model = KNeighborsClassifier
        elif task_type == "regression":
            ML_model = KNeighborsRegressor
    elif(model == "MLP"):
        hidden_layer = [
            model_cfg['max_value_hidden_layer_1'], model_cfg['min_value_hidden_layer_1']]
        hidden_layer_2 = [
            model_cfg['max_value_hidden_layer_2'], model_cfg['min_value_hidden_layer_2']]
        alpha = [model_cfg['max_value_alpha'],
                 model_cfg['min_value_alpha']]
        learning_rate_init = [
            model_cfg['max_value_learning_rate_init'], model_cfg['min_value_learning_rate_init']]
        max_iter = [model_cfg['max_value_max_iter'],
                    model_cfg['min_value_max_iter']]
        tol = [model_cfg['max_value_tol'], model_cfg['min_value_tol']]
        beta = [model_cfg['max_value_beta_1'], model_cfg['min_value_beta_1']]
        beta_2 = [model_cfg['max_value_beta_2'], model_cfg['min_value_beta_2']]
        n_iter_no_change = [
            model_cfg['max_value_n_iter_no_change'], model_cfg['min_value_n_iter_no_change']]
        activation = [model_cfg['max_value_activation'],
                      model_cfg['min_value_activation']]
        solver = [model_cfg['max_value_solver'],
                  model_cfg['min_value_solver']]
        learning_rate = [
            model_cfg['max_value_learning_rate'], model_cfg['min_value_learning_rate']]

        max_min = [hidden_layer, hidden_layer_2, alpha, learning_rate_init, max_iter,
                   tol, beta, beta_2, n_iter_no_change, activation, solver, learning_rate]
        if task_type == "classification":
            ML_model = MLPClassifier
        elif task_type == "regression":
            ML_model = MLPRegressor

    elif(model == "SVM"):
        c = [model_cfg['max_value_c'], model_cfg['min_value_c']]
        tol = [model_cfg['max_value_tol'], model_cfg['min_value_tol']]
        max_iter = [model_cfg['max_value_max_iter'],
                    model_cfg['min_value_max_iter']]
        gamma = [model_cfg['max_value_gamma'],
                 model_cfg['min_value_gamma']]
        kernal = [model_cfg['max_value_kernal'],
                  model_cfg['min_value_kernal']]
        max_min = [c, tol, max_iter, gamma, kernal]
        if task_type == "classification":
            ML_model = SVC
        elif task_type == "regression":
            ML_model = SVR
    return max_min, ML_model


def model_predict(model, ML_model, best_parameter, class_param, X, y, task_type, folder):
    for index, class_num in enumerate(class_param['class_number']):
        count = 1
        for class_name in class_param['class_name'][index]:
            if(count-1 < best_parameter[class_num] <= count):
                best_parameter[class_num] = class_name
                break
            else:
                count += 1
    if(model == "KNN"):
        predictor = ML_model(n_neighbors=best_parameter[0], leaf_size=best_parameter[1], metric=best_parameter[4],  weights=best_parameter[2],
                             algorithm=best_parameter[3], n_jobs=-1)
    elif(model == "MLP"):
        predictor = ML_model(hidden_layer_sizes=[best_parameter[0], best_parameter[1]],  alpha=best_parameter[2], learning_rate_init=best_parameter[3], max_iter=best_parameter[4], tol=best_parameter[5],
                             beta_1=best_parameter[6], beta_2=best_parameter[7], n_iter_no_change=best_parameter[8], activation=best_parameter[9], solver=best_parameter[10], learning_rate=best_parameter[11])
    elif(model == "SVM"):
        predictor = ML_model(C=best_parameter[0], tol=best_parameter[1], max_iter=best_parameter[2],
                             gamma=best_parameter[3], cache_size=1000, kernel=best_parameter[4])
    cofusion_model = predictor
    predictor.fit(X, y)
    pre = predictor.predict(X)
    predict_data = pd.concat(
        [X, pd.DataFrame(y),  pd.DataFrame(pre, columns=['predict'])], axis=1)
    if task_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)
        cofusion_model.fit(X_train, y_train)
        predictions = cofusion_model.predict(X_test)
        cm = confusion_matrix(y_test, predictions,
                              labels=cofusion_model.classes_)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=cofusion_model.classes_)
        disp.plot()
        plt.savefig(f'{folder}/{model}_confusion_matrix')
    return predict_data
