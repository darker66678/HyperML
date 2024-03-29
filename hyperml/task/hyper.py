import pandas as pd
import json
import datetime
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn.metrics
import matplotlib.pyplot as plt
import plotly.express as px
import sys
sys.path.append("..")
from algo.PSO import PSO
from algo.VOA import VOA
from algo.RANDOM import RANDOM_SEARCH
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def algo_MLconfig(model, task_type, model_cfg):
    if(model == "KNN"):
        neighbors = [model_cfg["neighbors"]["max"],
                     model_cfg["neighbors"]["min"]]
        leaf_size = [model_cfg["leaf_size"]["max"],
                     model_cfg["leaf_size"]["min"]]
        weights = [model_cfg["weights"]["max"],
                   model_cfg["weights"]["min"]]
        algorithm = [model_cfg["algorithm"]["max"],
                     model_cfg["algorithm"]["min"]]
        metric = [model_cfg["metric"]["max"],
                  model_cfg["metric"]["min"]]
        max_min = [neighbors, leaf_size, weights, algorithm, metric]
        if task_type == "classification":
            ML_model = KNeighborsClassifier
        elif task_type == "regression":
            ML_model = KNeighborsRegressor
    elif(model == "MLP"):
        hidden_layer_1 = [model_cfg['hidden_layer_1']
                          ['max'], model_cfg['hidden_layer_1']['min']]
        hidden_layer_2 = [model_cfg['hidden_layer_2']
                          ['max'], model_cfg['hidden_layer_2']['min']]
        alpha = [model_cfg['alpha']["max"],
                 model_cfg['alpha']["min"]]
        learning_rate_init = [
            model_cfg['learning_rate_init']['max'], model_cfg['learning_rate_init']["min"]]
        max_iter = [model_cfg['max_iter']["max"],
                    model_cfg['max_iter']["min"]]
        tol = [model_cfg['tol']["max"], model_cfg['tol']["min"]]
        beta_1 = [model_cfg['beta_1']["max"], model_cfg['beta_1']["min"]]
        beta_2 = [model_cfg['beta_2']["max"], model_cfg['beta_2']["min"]]
        n_iter_no_change = [
            model_cfg['n_iter_no_change']["max"], model_cfg['n_iter_no_change']["min"]]
        activation = [model_cfg['activation']["max"],
                      model_cfg['activation']["min"]]
        solver = [model_cfg['solver']["max"],
                  model_cfg['solver']["min"]]
        learning_rate = [
            model_cfg['learning_rate']["max"], model_cfg['learning_rate']["min"]]

        max_min = [hidden_layer_1, hidden_layer_2, alpha, learning_rate_init, max_iter,
                   tol, beta_1, beta_2, n_iter_no_change, activation, solver, learning_rate]
        if task_type == "classification":
            ML_model = MLPClassifier
        elif task_type == "regression":
            ML_model = MLPRegressor

    elif(model == "SVM"):
        c = [model_cfg['c']["max"], model_cfg['c']["min"]]
        tol = [model_cfg['tol']["max"], model_cfg['tol']["min"]]
        max_iter = [model_cfg['max_iter']["max"],
                    model_cfg['max_iter']["min"]]
        gamma = [model_cfg['gamma']["max"],
                 model_cfg['gamma']["min"]]
        kernal = [model_cfg['kernal']["max"],
                  model_cfg['kernal']["min"]]
        max_min = [c, tol, max_iter, gamma, kernal]
        if task_type == "classification":
            ML_model = SVC
        elif task_type == "regression":
            ML_model = SVR

    elif(model == "RF"):
        n_estimators = [model_cfg['n_estimators']
                        ["max"], model_cfg['n_estimators']["min"]]
        criterion = [model_cfg['criterion']["max"],
                     model_cfg['criterion']["min"]]
        max_depth = [model_cfg['max_depth']["max"],
                     model_cfg['max_depth']["min"]]
        min_samples_split = [model_cfg['min_samples_split']["max"],
                             model_cfg['min_samples_split']["min"]]
        min_samples_leaf = [model_cfg['min_samples_leaf']["max"],
                            model_cfg['min_samples_leaf']["min"]]
        max_features = [model_cfg['max_features']["max"],
                        model_cfg['max_features']["min"]]
        max_min = [n_estimators, criterion, max_depth,
                   min_samples_split, min_samples_leaf, max_features]
        if task_type == "classification":
            ML_model = RandomForestClassifier
        elif task_type == "regression":
            ML_model = RandomForestRegressor

    elif(model == "ADA"):
        n_estimators = [model_cfg['n_estimators']
                        ["max"], model_cfg['n_estimators']["min"]]
        learning_rate = [model_cfg['learning_rate']["max"],
                         model_cfg['learning_rate']["min"]]
        if(task_type == "classification"):
            algorithm = [model_cfg['algorithm']["max"],
                        model_cfg['algorithm']["min"]]
        elif(task_type == "regression"):
            loss = [model_cfg['loss']["max"],
                        model_cfg['loss']["min"]]
            algorithm = loss
        criterion = [model_cfg['criterion']["max"],
                     model_cfg['criterion']["min"]]
        max_depth = [model_cfg['max_depth']["max"],
                     model_cfg['max_depth']["min"]]
        min_samples_split = [model_cfg['min_samples_split']["max"],
                             model_cfg['min_samples_split']["min"]]
        min_samples_leaf = [model_cfg['min_samples_leaf']["max"],
                            model_cfg['min_samples_leaf']["min"]]
        max_features = [model_cfg['max_features']["max"],
                        model_cfg['max_features']["min"]]
        max_min = [n_estimators, learning_rate, algorithm, criterion, max_depth,
                   min_samples_split, min_samples_leaf, max_features]
        if task_type == "classification":
            ML_model = [AdaBoostClassifier, DecisionTreeClassifier]
        elif task_type == "regression":
            ML_model = [AdaBoostRegressor, DecisionTreeRegressor]

    elif(model == "XGBoost"):
        n_estimators = [model_cfg['n_estimators']
                        ["max"], model_cfg['n_estimators']["min"]]
        max_depth = [model_cfg['max_depth']
                     ["max"], model_cfg['max_depth']["min"]]
        max_leaves = [model_cfg['max_leaves']
                      ["max"], model_cfg['max_leaves']["min"]]
        grow_policy = [model_cfg['grow_policy']
                       ["max"], model_cfg['grow_policy']["min"]]
        learning_rate = [model_cfg['learning_rate']
                         ["max"], model_cfg['learning_rate']["min"]]
        booster = [model_cfg['booster']["max"], model_cfg['booster']["min"]]
        tree_method = [model_cfg['tree_method']
                       ["max"], model_cfg['tree_method']["min"]]
        gamma = [model_cfg['gamma']["max"], model_cfg['gamma']["min"]]
        min_child_weight = [model_cfg['min_child_weight']
                            ["max"], model_cfg['min_child_weight']["min"]]
        max_delta_step = [model_cfg['max_delta_step']
                          ["max"], model_cfg['max_delta_step']["min"]]
        subsample = [model_cfg['subsample']
                     ["max"], model_cfg['subsample']["min"]]
        colsample_bytree = [model_cfg['colsample_bytree']
                            ["max"], model_cfg['colsample_bytree']["min"]]
        colsample_bylevel = [model_cfg['colsample_bylevel']
                             ["max"], model_cfg['colsample_bylevel']["min"]]
        colsample_bynode = [model_cfg['colsample_bynode']
                            ["max"], model_cfg['colsample_bynode']["min"]]
        gammreg_alphaa = [model_cfg['reg_alpha']
                          ["max"], model_cfg['reg_alpha']["min"]]
        importance_type = [model_cfg['importance_type']
                           ["max"], model_cfg['importance_type']["min"]]
        max_min = [n_estimators, max_depth, max_leaves, grow_policy, learning_rate, booster, tree_method, gamma, min_child_weight,
                   max_delta_step, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, gammreg_alphaa, importance_type]
        if task_type == "classification":
            ML_model = xgboost.XGBClassifier
        elif task_type == "regression":
            ML_model = xgboost.XGBRegressor

    return max_min, ML_model

def plot_boxplot(data, boxpath):
    fig = px.box(data, y="test", title=boxpath[:-4].split("/")[-1])
    fig.write_image(boxpath)

def hyper(args, model_cfg, X, y, file, folder, task_type):
    if(args.algo == "PSO"):
        with open('./cfg/algo/PSO_config.json') as f:
            algo_cfg = json.load(f)
            print(f'PSO parameters: {algo_cfg}')
            logging.info(f'PSO config: {algo_cfg}')
        particle_num = algo_cfg['particle_num']
        particle_dim = len(model_cfg)
        iter_num = algo_cfg['iter_num']
        c1 = algo_cfg['c1']
        c2 = algo_cfg['c2']
        w = algo_cfg['w']

        starttime = datetime.datetime.now()
        pso = PSO(particle_num, particle_dim, iter_num, c1,
                  c2, w, model_cfg, X, y, file, args.model, folder, args.scoring, task_type, algo_MLconfig, args.k_fold)
        results, ML_model, best_parameter, class_param, param_name = pso.main()
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
        virus_dim = len(model_cfg)
        s_proportion = algo_cfg['s_proportion']
        strong_growth_rate = algo_cfg['strong_growth_rate']
        common_growth_rate = algo_cfg['common_growth_rate']
        total_virus_limit = algo_cfg['total_virus_limit']
        intensity = algo_cfg['intensity']
        starttime = datetime.datetime.now()
        voa = VOA(virus_num, virus_dim, strong_growth_rate,
                  common_growth_rate, s_proportion, total_virus_limit, intensity, model_cfg, X, y, args.model, args.scoring, task_type, folder, file, algo_MLconfig, args.k_fold)
        results, ML_model, best_parameter, class_param, param_name = voa.main()
        endtime = datetime.datetime.now()
        logging.info((f"time:{endtime - starttime} "))
        logging.info("-----------------------\n\n")
        print("time: ", (endtime - starttime))
        print("---------------------------")
        results.to_csv(
            f'{folder}/{args.algo}({args.model})_{file}.csv')

    elif(args.algo == "RANDOM"):
        with open(f'./cfg/algo/{args.algo}_config.json') as f:
            algo_cfg = json.load(f)
            print(f'RANDOM SERARCH config: {algo_cfg}')
            logging.info(f'RANDOM SERARCH config: {algo_cfg}')

        iter_num = algo_cfg['iter_num']
        starttime = datetime.datetime.now()
        results, ML_model, best_parameter, class_param, param_name = RANDOM_SEARCH(
            iter_num, model_cfg, X, y, args.model, args.scoring, task_type, folder, file, algo_MLconfig, args.k_fold)
        endtime = datetime.datetime.now()
        logging.info((f"time:{endtime - starttime} "))
        logging.info("-----------------------\n\n")
        print("time: ", (endtime - starttime))
        print("---------------------------")
        results.to_csv(
            f'{folder}/{args.algo}({args.model})_{file}.csv')

    else:
        raise (ValueError(f'Unknown algo:{args.algo}'))
        
    boxpath = f'{folder}/{args.algo}_{args.model}_{file}_box.jpg'
    plot_boxplot(results, boxpath)
    cv_pred_data = model_predict(ML_model, best_parameter, class_param,
                                 X, y, task_type, folder, param_name, args)
    cv_pred_data.to_csv(
        f'{folder}/{args.algo}({args.model})_{file}_cv_predict.csv')

    logging.info("finished!!!")
    print("finished!!!")
    

def model_predict(ML_model, parameter, class_param, X, y, task_type, folder, param_name, args):
    for index, class_num in enumerate(class_param):
        count = 1
        for class_name in param_name[class_num]:
            if(not(isinstance(parameter[class_num], str))):
                if(count-1 < parameter[class_num] <= count):
                    parameter[class_num] = class_name
                    break
                else:
                    count += 1
    model = args.model
    if(model == "KNN"):
        predictor = ML_model(n_neighbors=parameter[0], leaf_size=parameter[1], metric=parameter[4],  weights=parameter[2],
                             algorithm=parameter[3], n_jobs=-1)
    elif(model == "MLP"):
        predictor = ML_model(hidden_layer_sizes=[parameter[0], parameter[1]],  alpha=parameter[2], learning_rate_init=parameter[3], max_iter=parameter[4], tol=parameter[5],
                             beta_1=parameter[6], beta_2=parameter[7], n_iter_no_change=parameter[8], activation=parameter[9], solver=parameter[10], learning_rate=parameter[11])
    elif(model == "SVM"):
        predictor = ML_model(C=parameter[0], tol=parameter[1], max_iter=parameter[2],
                             gamma=parameter[3], cache_size=1000, kernel=parameter[4])
    elif(model == "RF"):
        predictor = ML_model(n_estimators=parameter[0],
                             max_depth=parameter[2], min_samples_split=parameter[3], min_samples_leaf=parameter[4], criterion=parameter[1], max_features=parameter[5], n_jobs=-1,)
    elif(model == "ADA"):
        if task_type == "classification":
            predictor = ML_model[0](n_estimators=parameter[0], learning_rate=parameter[1], algorithm=parameter[2], base_estimator=ML_model[1](
            criterion=parameter[3], max_depth=parameter[4], min_samples_split=parameter[5], min_samples_leaf=parameter[6], max_features=parameter[7]))
        elif task_type == "regression":
            predictor = ML_model[0](n_estimators=parameter[0], learning_rate=parameter[1], loss=parameter[2], base_estimator=ML_model[1](
            criterion=parameter[3], max_depth=parameter[4], min_samples_split=parameter[5], min_samples_leaf=parameter[6], max_features=parameter[7]))
        
    elif(model == "XGBoost"):
        predictor = ML_model(n_estimators=parameter[0], max_depth=parameter[1],
                             max_leaves=parameter[2], grow_policy=parameter[3], learning_rate=parameter[4], booster=parameter[5], tree_method=parameter[6], gamma=parameter[7], min_child_weight=parameter[8], max_delta_step=parameter[9], subsample=parameter[10], colsample_bytree=parameter[11], colsample_bylevel=parameter[12], colsample_bynode=parameter[13], reg_alpha=parameter[14], importance_type=parameter[15], n_jobs=-1)

    cv_pred, all_label = cross_validation(X, y, args, predictor, task_type)
    cv_pred_data = pd.concat([X, pd.DataFrame(y),  cv_pred], axis=1)
    eval_metrics(task_type, y, cv_pred)
    if args.confusion_m:
        if task_type == "classification":
            cm = confusion_matrix(
                y, cv_pred, labels=all_label)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=all_label)
            disp.plot()
            plt.savefig(f'{folder}/{model}_confusion_matrix')
            logging.info("Save confusio matrix!")
        else:
            logging.debug("Regression can't plot confusio matrix!")
    return cv_pred_data

def cross_validation(X, y, args, init_predictor, task_type):
    k_fold = args.k_fold
    unit = int(len(X)/k_fold)
    total_pred = pd.DataFrame()
    predictor = init_predictor
    for i in range(k_fold):
        if i == k_fold-1:
            train_x = X.drop(range(unit*i, len(X)))
            train_y = y.drop(range(unit*i, len(X)))
            test_x = X.iloc[unit*i:len(X), :]
        else:
            train_x = X.drop(range(unit*i, unit*(i+1)))
            train_y = y.drop(range(unit*i, unit*(i+1)))
            test_x = X.iloc[unit*i:unit*(i+1), :]
        predictor.fit(train_x, train_y)
        pred = pd.Series(predictor.predict(test_x))
        pred.index = test_x.index
        total_pred = pd.concat([total_pred, pred], axis=0)
    total_pred.columns = ['cv_predict']
    if(task_type == "classification"):
        all_label = predictor.classes_
    elif(task_type == "regression"):
        all_label = []
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return total_pred, all_label

def eval_metrics(task_type, y, cv_pred):
    if task_type == "regression":
        r2 = sklearn.metrics.r2_score(y, cv_pred)
        mse = sklearn.metrics.mean_squared_error(y, cv_pred)
        mae = sklearn.metrics.mean_absolute_error(y, cv_pred)
        logging.info("------ Cross validation metrics of Best model ------")
        logging.info(f'R2:{r2}')
        logging.info(f'MSE:{mse}')
        logging.info(f'MAE:{mae}')

    elif task_type == "classification":
        accuracy = sklearn.metrics.accuracy_score(y, cv_pred)
        f1 = sklearn.metrics.f1_score(y, cv_pred)
        recall = sklearn.metrics.recall_score(y, cv_pred, pos_label=1)
        precision = sklearn.metrics.precision_score(y, cv_pred)
        specificity = sklearn.metrics.recall_score(y, cv_pred, pos_label=0)
        logging.info("------ Cross validation metrics of Best model ------")
        logging.info(f'Accuracy:{accuracy}')
        logging.info(f'F1 Score:{f1}')
        logging.info(f'Precision:{precision}')
        logging.info(f'Recall:{recall}')
        logging.info(f'Specificity:{specificity}')
    logging.info("---------------------------------------------------")
