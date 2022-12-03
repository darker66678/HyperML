import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_validate
from sklearn.utils import parallel_backend
import logging
import xgboost as xg
from tqdm import tqdm
from sklearn.metrics import make_scorer, recall_score
specificity = make_scorer(recall_score, pos_label=0)


def ML_cross_validation(model, ML_model, current_params, class_particle, k_fold, scoring, X, y, params,task_type):
    if(model == "KNN"):
        with parallel_backend('threading'):
            knn = ML_model(
                n_neighbors=current_params[0], weights=class_particle[0], algorithm=class_particle[1], leaf_size=current_params[1], metric=class_particle[2], n_jobs=-1)
            cv_scores = cross_validate(
                knn, X, y, cv=k_fold, scoring=scoring, n_jobs=-1, return_train_score=True)
        params.append([current_params[0], current_params[1], class_particle[0],
                       class_particle[1], class_particle[2]])
    elif(model == "ADA"):
        with parallel_backend('threading'):
            if(task_type == "classfication"):
                ada = ML_model[0](n_estimators=current_params[0], learning_rate=current_params[1], algorithm=class_particle[0], base_estimator=ML_model[1](
                criterion=class_particle[1], max_depth=current_params[4], min_samples_split=current_params[5], min_samples_leaf=current_params[6], max_features=class_particle[2]))
            else:
                ada = ML_model[0](n_estimators=current_params[0], learning_rate=current_params[1], loss=class_particle[0], base_estimator=ML_model[1](
                criterion=class_particle[1], max_depth=current_params[4], min_samples_split=current_params[5], min_samples_leaf=current_params[6], max_features=class_particle[2]))
            cv_scores = cross_validate(
                ada, X, y, cv=k_fold, scoring=scoring, n_jobs=-1, return_train_score=True)
        params.append([current_params[0], current_params[1], class_particle[0],
                       class_particle[1], current_params[4], current_params[5], current_params[6], class_particle[2]])
    elif(model == "RF"):
        with parallel_backend('threading'):
            rf = ML_model(n_estimators=current_params[0], criterion=class_particle[0],
                          max_depth=current_params[2], min_samples_split=current_params[3], min_samples_leaf=current_params[4],
                          max_features=class_particle[1], n_jobs=-1)
            cv_scores = cross_validate(
                rf, X, y, cv=k_fold, scoring=scoring, n_jobs=-1, return_train_score=True)
        params.append([current_params[0], class_particle[0], current_params
                       [2], current_params[3], current_params[4], class_particle[1]])

    elif(model == "MLP"):
        with parallel_backend('threading'):
            mlp = ML_model(hidden_layer_sizes=[current_params[0], current_params[1]], activation=class_particle[0], solver=class_particle[1], alpha=current_params[2], learning_rate=class_particle[2], learning_rate_init=current_params[3], max_iter=current_params[4],  tol=current_params[5],
                           beta_1=current_params[6], beta_2=current_params[7],
                           n_iter_no_change=current_params[8])
            cv_scores = cross_validate(
                mlp, X, y, cv=k_fold, scoring=scoring, n_jobs=-1, return_train_score=True)
        params.append(
            [current_params[0], current_params[1],  current_params[2], current_params[3], current_params[4],
                current_params[5], current_params[6], current_params[7], current_params[8], class_particle[0], class_particle[1], class_particle[2]])

    elif(model == "SVM"):
        with parallel_backend('threading'):
            svm = ML_model(C=current_params[0], kernel=class_particle[0],  gamma=current_params
                           [3], tol=current_params[1], cache_size=1000, max_iter=current_params[2])
            cv_scores = cross_validate(
                svm, X, y, cv=k_fold, scoring=scoring, n_jobs=-1, return_train_score=True)
        params.append(
            [current_params[0], current_params[1], current_params[2], current_params[3], class_particle[0]])

    elif (model == "XGBoost"):
        with parallel_backend('threading'):
            xg_params = {"n_estimators": current_params[0], "max_depth": current_params[1],
                         "max_leaves": current_params[2], "grow_policy": class_particle[0], "learning_rate": current_params[4], "booster": class_particle[1], "tree_method": class_particle[2], "gamma": current_params[7], "min_child_weight": current_params[8], "max_delta_step": current_params[9], "subsample": current_params[10], "colsample_bytree": current_params[11], "colsample_bylevel": current_params[12], "colsample_bynode": current_params[13], "reg_alpha": current_params[14], "importance_type": class_particle[3], "n_jobs": -1}
            xg = ML_model(**xg_params)
            cv_scores = cross_validate(
                xg, X, y, cv=k_fold, scoring=scoring, n_jobs=-1, return_train_score=True)
        params.append([current_params[0], current_params[1],
                       current_params[2], class_particle[0], current_params[4], class_particle[1], class_particle[2], current_params[7], current_params[8], current_params[9], current_params[10], current_params[11], current_params[12], current_params[13], current_params[14], class_particle[3]])

    return params, cv_scores


def RANDOM_SEARCH(iter_num, model_cfg, X, y, model, scoring, task_type, folder, file, algo_MLconfig, k_fold):
    int_param = []
    class_param = []
    param_name = []
    for index, i in enumerate(model_cfg.keys()):
        if model_cfg[i]['int']:
            int_param.append(index)
        if model_cfg[i]['class']:
            class_param.append(index)
            param_name.append(model_cfg[i]["class_name"])
        else:
            param_name.append([])

    max_min, ML_model = algo_MLconfig(model, task_type, model_cfg)

    test = []
    train = []
    params = []

    for i in tqdm(range(iter_num), ncols=80):
        current_params = []
        for param_num in range(len(max_min)):
            if param_num in int_param:
                current_params.append(random.randint(
                    max_min[param_num][1], max_min[param_num][0]))
            else:
                current_params.append(
                    random.uniform(max_min[param_num][1], max_min[param_num][0]))

        class_particle = []
        for index, class_num in enumerate(class_param):
            count = 1
            for class_name in param_name[class_num]:
                if(count-1 < current_params[class_num] <= count):
                    class_particle.append(class_name)
                    break
                else:
                    count += 1
        params, cv_scores = ML_cross_validation(
            model, ML_model, current_params, class_particle, k_fold, scoring, X, y, params, task_type)
        mean_test_score = cv_scores['test_score'].mean()
        logging.info(f'{len(params)}, {mean_test_score}')
        test.append(mean_test_score)  # test data
        train.append(cv_scores['train_score'].mean())  # train data

    data_params = pd.DataFrame(params)
    data_params.columns = model_cfg.keys()
    results = pd.concat(
        [pd.DataFrame(train, columns=["train"]), pd.DataFrame(test, columns=["test"]), data_params], axis=1)
    best_index = test.index(max(test))
    best_parameter = params[best_index]
    logging.info(
        (f'best_parameter:{best_parameter}, {list(model_cfg.keys())}, best_fitness:{test[best_index]}'))
    return results, ML_model, best_parameter, class_param, param_name
