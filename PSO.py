

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import numpy as np
import random
from sklearn.utils import parallel_backend
import logging
from tqdm import tqdm, trange
from sklearn.metrics import make_scorer, recall_score
specificity = make_scorer(recall_score, pos_label=0)


class PSO(object):
    def __init__(self, particle_num, particle_dim, iter_num, c1, c2, w, model_cfg, X, y, file, model, folder, matrix):
        self.folder = folder
        self.model = model
        self.file = file
        self.X = X
        self.y = y
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.count = 0

        if matrix == 'specificity':
            self.matrix = specificity
        else:
            self.matrix = matrix

        if(model == "KNN"):
            self.max_value_neighbors = model_cfg["max_value_neighbors"]
            self.min_value_neighbors = model_cfg["min_value_neighbors"]
            self.max_value_leaf_size = model_cfg["max_value_leaf_size"]
            self.min_value_leaf_size = model_cfg["min_value_leaf_size"]
            self.max_value_weights = model_cfg["max_value_weights"]
            self.min_value_weights = model_cfg["min_value_weights"]
            self.max_value_algorithm = model_cfg["max_value_algorithm"]
            self.min_value_algorithm = model_cfg["min_value_algorithm"]
            self.max_value_metric = model_cfg["max_value_metric"]
            self.min_value_metric = model_cfg["min_value_metric"]
            if model_cfg["type"] == "classification":
                self.sklearn = KNeighborsClassifier
            elif model_cfg["type"] == "regression":
                self.sklearn = KNeighborsRegressor

        elif(model == "MLP"):
            self.max_value_hidden_layer_1 = model_cfg['max_value_hidden_layer_1']
            self.min_value_hidden_layer_1 = model_cfg['min_value_hidden_layer_1']
            self.max_value_hidden_layer_2 = model_cfg['max_value_hidden_layer_2']
            self.min_value_hidden_layer_2 = model_cfg['min_value_hidden_layer_2']
            self.max_value_alpha = model_cfg['max_value_alpha']
            self.min_value_alpha = model_cfg['min_value_alpha']
            self.max_value_learning_rate_init = model_cfg['max_value_learning_rate_init']
            self.min_value_learning_rate_init = model_cfg['min_value_learning_rate_init']
            self.max_value_max_iter = model_cfg['max_value_max_iter']
            self.min_value_max_iter = model_cfg['min_value_max_iter']
            self.max_value_tol = model_cfg['max_value_tol']
            self.min_value_tol = model_cfg['min_value_tol']
            self.max_value_beta_1 = model_cfg['max_value_beta_1']
            self.min_value_beta_1 = model_cfg['min_value_beta_1']
            self.max_value_beta_2 = model_cfg['max_value_beta_1']
            self.min_value_beta_2 = model_cfg['min_value_beta_2']
            self.max_value_n_iter_no_change = model_cfg['max_value_n_iter_no_change']
            self.min_value_n_iter_no_change = model_cfg['min_value_n_iter_no_change']
            self.max_value_activation = model_cfg['max_value_activation']
            self.min_value_activation = model_cfg['min_value_activation']
            self.max_value_solver = model_cfg['max_value_solver']
            self.min_value_solver = model_cfg['min_value_solver']
            self.max_value_learning_rate = model_cfg['max_value_learning_rate']
            self.min_value_learning_rate = model_cfg['min_value_learning_rate']
            if model_cfg["type"] == "classification":
                self.sklearn = MLPClassifier
            elif model_cfg["type"] == "regression":
                self.sklearn = MLPRegressor

        elif(model == "SVM"):
            self.max_value_c = model_cfg['max_value_c']
            self.min_value_c = model_cfg['min_value_c']
            self.max_value_tol = model_cfg['max_value_tol']
            self.min_value_tol = model_cfg['min_value_tol']
            self.max_value_max_iter = model_cfg['max_value_max_iter']
            self.min_value_max_iter = model_cfg['min_value_max_iter']
            self.max_value_gamma = model_cfg['max_value_gamma']
            self.min_value_gamma = model_cfg['min_value_gamma']
            self.max_value_kernal = model_cfg['max_value_kernal']
            self.min_value_kernal = model_cfg['min_value_kernal']
            if model_cfg["type"] == "classification":
                self.sklearn = SVC
            elif model_cfg["type"] == "regression":
                self.sklearn = SVR

# 2.1 粒子群初始化
    def swarm_origin(self):
        particle_loc = []
        particle_dir = []
        for i in range(self.particle_num):
            tmp1 = []
            tmp2 = []
            for j in range(self.particle_dim):
                a = random.random()
                if(self.model == "KNN"):
                    if (j == 0):
                        tmp1.append(round(
                            a * (self.max_value_neighbors - self.min_value_neighbors) + self.min_value_neighbors))
                        tmp2.append(
                            random.uniform(-self.max_value_neighbors/3, self.max_value_neighbors/3))
                    elif (j == 1):
                        tmp1.append(round(
                            (a * (self.max_value_leaf_size - self.min_value_leaf_size) + self.min_value_leaf_size)))
                        tmp2.append(
                            random.uniform(-self.max_value_leaf_size/3, self.max_value_leaf_size/3))
                    elif(j == 2):
                        tmp1.append(round(
                            (a * (self.max_value_weights - self.min_value_weights) + self.min_value_weights)))
                        tmp2.append(
                            random.uniform(-self.max_value_weights/3, self.max_value_weights/3))
                    elif(j == 3):
                        tmp1.append(round(
                            (a * (self.max_value_algorithm - self.min_value_algorithm) + self.min_value_algorithm)))
                        tmp2.append(
                            random.uniform(-self.max_value_algorithm/3, self.max_value_algorithm/3))
                    elif(j == 4):
                        tmp1.append(round(
                            (a * (self.max_value_metric - self.min_value_metric) + self.min_value_metric)))
                        tmp2.append(
                            random.uniform(-self.max_value_metric/3, self.max_value_metric/3))

                elif(self.model == "MLP"):
                    if (j == 0):
                        tmp1.append(round(a * (self.max_value_hidden_layer_1 -
                                               self.min_value_hidden_layer_1) + self.min_value_hidden_layer_1))
                        tmp2.append(
                            random.uniform(-self.max_value_hidden_layer_1/3, self.max_value_hidden_layer_1/3))
                    elif (j == 1):
                        tmp1.append(round(a * (self.max_value_hidden_layer_2 -
                                               self.min_value_hidden_layer_2) + self.min_value_hidden_layer_2))
                        tmp2.append(
                            random.uniform(-self.max_value_hidden_layer_2/3, self.max_value_hidden_layer_2/3))
                    elif (j == 2):
                        tmp1.append(round(
                            (a * (self.max_value_alpha - self.min_value_alpha) + self.min_value_alpha), 5))
                        tmp2.append(
                            random.uniform(-self.max_value_alpha/3, self.max_value_alpha/3))
                    elif(j == 3):
                        tmp1.append(round((a * (self.max_value_learning_rate_init -
                                                self.min_value_learning_rate_init) + self.min_value_learning_rate_init), 4))
                        tmp2.append(random.uniform(-self.max_value_learning_rate_init /
                                                   3, self.max_value_learning_rate_init/3))
                    elif(j == 4):
                        tmp1.append(round(a * (self.max_value_max_iter -
                                               self.min_value_max_iter) + self.min_value_max_iter))
                        tmp2.append(
                            random.uniform(-self.max_value_max_iter/3, self.max_value_max_iter/3))
                    elif(j == 5):
                        tmp1.append(
                            round((a * (self.max_value_tol - self.min_value_tol) + self.min_value_tol), 5))
                        tmp2.append(
                            random.uniform(-self.max_value_tol/3, self.max_value_tol/3))
                    elif(j == 6):
                        tmp1.append(round(
                            (a * (self.max_value_beta_1 - self.min_value_beta_1) + self.min_value_beta_1), 4))
                        tmp2.append(
                            random.uniform(-self.max_value_beta_1/3, self.max_value_beta_1/3))
                    elif(j == 7):
                        tmp1.append(round(
                            (a * (self.max_value_beta_2 - self.min_value_beta_2) + self.min_value_beta_2), 4))
                        tmp2.append(
                            random.uniform(-self.max_value_beta_2/3, self.max_value_beta_2/3))
                    elif(j == 8):
                        tmp1.append(round((a * (self.max_value_n_iter_no_change -
                                                self.min_value_n_iter_no_change) + self.min_value_n_iter_no_change)))
                        tmp2.append(
                            random.uniform(-self.max_value_n_iter_no_change/3, self.max_value_n_iter_no_change/3))
                    elif(j == 9):
                        tmp1.append(round((a * (self.max_value_activation -
                                                self.min_value_activation) + self.min_value_activation)))
                        tmp2.append(
                            random.uniform(-self.max_value_activation/3, self.max_value_activation/3))
                    elif(j == 10):
                        tmp1.append(round(
                            (a * (self.max_value_solver - self.min_value_solver) + self.min_value_solver)))
                        tmp2.append(
                            random.uniform(-self.max_value_solver/3, self.max_value_solver/3))
                    elif(j == 11):
                        tmp1.append(round((a * (self.max_value_learning_rate -
                                                self.min_value_learning_rate) + self.min_value_learning_rate)))
                        tmp2.append(
                            random.uniform(-self.max_value_learning_rate/3, self.max_value_learning_rate/3))

                elif(self.model == "SVM"):
                    if (j == 0):
                        tmp1.append(
                            (a * (self.max_value_c - self.min_value_c) + self.min_value_c))
                        tmp2.append(
                            random.uniform(-self.max_value_c/3, self.max_value_c/3))
                    elif (j == 1):
                        tmp1.append(
                            (a * (self.max_value_tol - self.min_value_tol) + self.min_value_tol))
                        tmp2.append(
                            random.uniform(-self.max_value_tol/3, self.max_value_tol/3))
                    elif(j == 2):
                        tmp1.append(
                            (a * (self.max_value_max_iter - self.min_value_max_iter) + self.min_value_max_iter))
                        tmp2.append(
                            random.uniform(-self.max_value_max_iter/3, self.max_value_max_iter/3))
                    elif(j == 3):
                        tmp1.append(
                            (a * (self.max_value_gamma - self.min_value_gamma) + self.min_value_gamma))
                        tmp2.append(
                            random.uniform(-self.max_value_gamma/3, self.max_value_gamma/3))
                    elif(j == 4):
                        tmp1.append(round(
                            (a * (self.max_value_kernal - self.min_value_kernal) + self.min_value_kernal)))
                        tmp2.append(
                            random.uniform(-self.max_value_kernal/3, self.max_value_kernal/3))
            particle_loc.append(tmp1)
            particle_dir.append(tmp2)

        return particle_loc, particle_dir

    def fitness(self, particle_loc, test, train, parms):
        fitness_value = []
        for i in range(self.particle_num):
            if(self.model == "KNN"):
                if(particle_loc[i][2] == 1):
                    weights = "uniform"
                elif(particle_loc[i][2] == 2):
                    weights = "distance"

                if(particle_loc[i][3] == 1):
                    algorithm = "ball_tree"
                elif(particle_loc[i][3] == 2):
                    algorithm = "kd_tree"
                elif(particle_loc[i][3] == 3):
                    algorithm = "brute"
                elif(particle_loc[i][3] == 4):
                    algorithm = "auto"

                if(particle_loc[i][4] == 1):
                    metric = "euclidean"
                elif(particle_loc[i][4] == 2):
                    metric = "manhattan"
                elif(particle_loc[i][4] == 3):
                    metric = "chebyshev"

                with parallel_backend('threading'):
                    knn = self.sklearn(n_neighbors=particle_loc[i][0], weights=weights, algorithm=algorithm,
                                       leaf_size=particle_loc[i][1], metric=metric, metric_params=None, n_jobs=-1,)
                    cv_scores = cross_validate(
                        knn, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                parms.append([particle_loc[i][0], particle_loc[i]
                              [1], weights, algorithm, metric])

            elif(self.model == "MLP"):
                if(particle_loc[i][9] == 1):
                    activation = "identity"
                elif(particle_loc[i][9] == 2):
                    activation = "logistic"
                elif(particle_loc[i][9] == 3):
                    activation = "tanh"
                elif(particle_loc[i][9] == 4):
                    activation = "relu"

                if(particle_loc[i][10] == 1):
                    solver = "lbfgs"
                elif(particle_loc[i][10] == 2):
                    solver = "sgd"
                elif(particle_loc[i][10] == 3):
                    solver = "adam"

                if(particle_loc[i][11] == 1):
                    learning_rate = "constant"
                elif(particle_loc[i][11] == 2):
                    learning_rate = "invscaling"
                elif(particle_loc[i][11] == 3):
                    learning_rate = "adaptive"

                with parallel_backend('threading'):
                    mlp = self.sklearn(hidden_layer_sizes=[particle_loc[i][0], particle_loc[i][1]], activation=activation, solver=solver, alpha=particle_loc[i][2], batch_size='auto', learning_rate=learning_rate, learning_rate_init=particle_loc[i][3], max_iter=particle_loc[i]
                                       [4], shuffle=True, random_state=None, tol=particle_loc[i][5], verbose=False, warm_start=False, nesterovs_momentum=True, early_stopping=False, beta_1=particle_loc[i][6], beta_2=particle_loc[i][7], epsilon=1e-08, n_iter_no_change=particle_loc[i][8])
                    cv_scores = cross_validate(
                        mlp, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                parms.append([particle_loc[i][0], particle_loc[i][1], particle_loc[i][2], particle_loc[i][3], particle_loc[i][4],
                              particle_loc[i][5], particle_loc[i][6], particle_loc[i][7], particle_loc[i][8], activation, solver, learning_rate])

            elif (self.model == "SVM"):
                if(particle_loc[i][4] == 1):
                    kernel = "linear"
                elif(particle_loc[i][4] == 2):
                    kernel = "poly"
                elif(particle_loc[i][4] == 3):
                    kernel = "rbf"
                elif(particle_loc[i][4] == 4):
                    kernel = "sigmoid"

                with parallel_backend('threading'):
                    svm = self.sklearn(C=particle_loc[i][0], tol=particle_loc[i][1],
                                       max_iter=particle_loc[i][2], gamma=particle_loc[i][3], cache_size=1000)
                    cv_scores = cross_validate(
                        svm, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                parms.append([particle_loc[i][0], particle_loc[i][1],
                              particle_loc[i][2], particle_loc[i][3], kernel])

            self.count += 1
            logging.info(
                f"number {self.count} , {cv_scores['test_score'].mean()}")

            fitness_value.append(cv_scores['test_score'].mean())
            test.append(cv_scores['test_score'].mean())
            train.append(cv_scores['train_score'].mean())
        current_fitness = 0.0
        current_parameter = []
        for i in range(self.particle_num):
            if current_fitness < fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]

        return fitness_value, current_fitness, current_parameter

    def update(self, particle_loc, particle_dir, gbest_parameter, pbest_parameters):
        for i in range(self.particle_num):
            a1 = [x * self.w for x in particle_dir[i]]
            a2 = [y * self.c1 * random.random()
                  for y in list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
            a3 = [z * self.c2 * random.random()
                  for z in list(np.array(gbest_parameter) - np.array(particle_loc[i]))]
            particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
            particle_loc[i] = list(
                np.array(particle_loc[i]) + np.array(particle_dir[i]))
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        value = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)

        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                if(self.model == "KNN"):
                    if (j == 0):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_neighbors - self.min_value_neighbors)+self.min_value_neighbors))
                    elif (j == 1):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_leaf_size - self.min_value_leaf_size) + self.min_value_leaf_size))
                    elif(j == 2):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_weights - self.min_value_weights) + self.min_value_weights))
                    elif(j == 3):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_algorithm - self.min_value_algorithm) + self.min_value_algorithm))
                    elif(j == 4):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_metric - self.min_value_metric) + self.min_value_metric))

                elif(self.model == "MLP"):
                    if (j == 0):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_hidden_layer_1 - self.min_value_hidden_layer_1) + self.min_value_hidden_layer_1))
                    elif (j == 1):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_hidden_layer_2 - self.min_value_hidden_layer_2) + self.min_value_hidden_layer_2))
                    elif (j == 2):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_alpha - self.min_value_alpha) + self.min_value_alpha), 5)
                    elif(j == 3):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_learning_rate_init - self.min_value_learning_rate_init) + self.min_value_learning_rate_init), 4)
                    elif(j == 4):
                        particle_loc[i][j] = round((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_max_iter - self.min_value_max_iter) + self.min_value_max_iter)
                    elif(j == 5):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_tol - self.min_value_tol) + self.min_value_tol), 5)
                    elif(j == 6):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_beta_1 - self.min_value_beta_1) + self.min_value_beta_1), 4)
                    elif(j == 7):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_beta_2 - self.min_value_beta_2) + self.min_value_beta_2), 4)
                    elif(j == 8):
                        particle_loc[i][j] = round((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_n_iter_no_change - self.min_value_n_iter_no_change) + self.min_value_n_iter_no_change)
                    elif(j == 9):
                        particle_loc[i][j] = round((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_activation - self.min_value_activation) + self.min_value_activation)
                    elif(j == 10):
                        particle_loc[i][j] = round((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_solver - self.min_value_solver) + self.min_value_solver)
                    elif(j == 11):
                        particle_loc[i][j] = round((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_learning_rate - self.min_value_learning_rate) + self.min_value_learning_rate)

                elif(self.model == "SVM"):
                    if (j == 0):
                        particle_loc[i][j] = ((particle_loc[i][j] - value[j][1])/(
                            value[j][0] - value[j][1]) * (self.max_value_c - self.min_value_c) + self.min_value_c)
                    elif (j == 1):
                        particle_loc[i][j] = ((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_tol - self.min_value_tol) + self.min_value_tol)
                    elif(j == 2):
                        particle_loc[i][j] = ((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_max_iter - self.min_value_max_iter) + self.min_value_max_iter)
                    elif(j == 3):
                        particle_loc[i][j] = ((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_gamma - self.min_value_gamma) + self.min_value_gamma)
                    elif(j == 4):
                        particle_loc[i][j] = round(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                            self.max_value_kernal - self.min_value_kernal) + self.min_value_kernal))

        return particle_loc, particle_dir

    def plot(self, results, results_ave, good_fitness):
        X = []
        Y = []
        Z = []
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
            Z.append(results_ave[i])
        plt.figure()
        plt.plot(X, Y, label='best fitness', color='blue')
        plt.plot(X, Z, label='average fitness each iteration',
                 color='green')
        plt.plot(X, good_fitness, label="best fitness each iteration",
                 color='orange')
        plt.xlabel('Number of iteration', size=15)
        plt.ylabel(f'{self.matrix} Score', size=15)
        plt.title(f'PSO tunes {self.model} hyperparameter optimization')
        plt.legend(loc='best')
        plt.savefig(f'{self.folder}/PSO_{self.model}_{self.file}')

    def main(self):

        results = []
        results_ave = []
        test = []
        train = []
        parms = []
        good_fitness = []
        best_fitness = 0.0
        particle_loc, particle_dir = self.swarm_origin()
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        pbest_parameters = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameters.append(tmp1)
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(0.0)

        print("PSO start!!")
        for i in tqdm(range(self.iter_num), ncols=80):
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(
                particle_loc, test, train, parms)
            current_fitness_sum = 0
            nan_number = 0
            for j in range(self.particle_num):
                if current_fitness_value[j] > fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
                if(np.isnan(current_fitness_value[j])):
                    nan_number = nan_number+1
                else:
                    current_fitness_sum = current_fitness_sum + \
                        current_fitness_value[j]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter
            logging.info(
                f"iteration is :,{i+1},Best parameters:{gbest_parameter},Best fitness: {best_fitness}")
            #print(f"iteration is :,{i+1},Best parameters:{gbest_parameter},Best fitness: {best_fitness}")

            results.append(best_fitness)
            good_fitness.append(current_best_fitness)
            # calculate mean, it will delete number of nan value
            results_ave.append(current_fitness_sum /
                               (self.particle_num-nan_number))
            fitness_value = current_fitness_value
            particle_loc, particle_dir = self.update(
                particle_loc, particle_dir, gbest_parameter, pbest_parameters)
        results.sort()
        self.plot(results, results_ave, good_fitness)

        logging.info(f'Final parameters are :{gbest_parameter}')
        print('Final parameters are :', gbest_parameter)
        results = pd.concat(
            [pd.Series(train), pd.Series(test), pd.Series(parms)], axis=1)
        results.columns = ['train', 'test', 'parameters']
        return results
