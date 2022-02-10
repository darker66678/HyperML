
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn.utils import parallel_backend
import logging
from sklearn.metrics import make_scorer, recall_score
specificity = make_scorer(recall_score, pos_label=0)


class VOA(object):
    def __init__(self, virus_num, virus_dim, strong_growth_rate, common_growth_rate, s_proportion, total_virus_limit,
                 intensity, model_cfg, X, y, model, matrix):

        self.virus_num = virus_num
        self.virus_dim = virus_dim
        self.s_proportion = s_proportion
        self.strong_growth_rate = strong_growth_rate  # 通常设为
        self.common_growth_rate = common_growth_rate  # 通常设为
        self.total_virus_limit = total_virus_limit
        self.intensity = intensity
        self.int_parameter = model_cfg['int_parameter']
        self.count = self.virus_num-1
        self.X = X
        self.y = y
        self.model = model
        if matrix == 'specificity':
            self.matrix = specificity
        else:
            self.matrix = matrix
        # -------------------------------------------------------------
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
            self.max_min = [neighbors, leaf_size, weights, algorithm, metric]
            if model_cfg["type"] == "classification":
                self.sklearn = KNeighborsClassifier
            elif model_cfg["type"] == "regression":
                self.sklearn = KNeighborsRegressor

        elif(model == "MLP"):
            hidden_layer = [
                model_cfg['max_value_hidden_layer'], model_cfg['min_value_hidden_layer']]
            hidden_layer_2 = [
                model_cfg['max_value_hidden_layer'], model_cfg['min_value_hidden_layer']]
            alpha = [model_cfg['max_value_alpha'],
                     model_cfg['min_value_alpha']]
            learning_rate_init = [
                model_cfg['max_value_learning_rate_init'], model_cfg['min_value_learning_rate_init']]
            max_iter = [model_cfg['max_value_max_iter'],
                        model_cfg['min_value_max_iter']]
            tol = [model_cfg['max_value_tol'], model_cfg['min_value_tol']]
            beta = [model_cfg['max_value_beta'], model_cfg['min_value_beta']]
            beta_2 = [model_cfg['max_value_beta'], model_cfg['min_value_beta']]
            n_iter_no_change = [
                model_cfg['max_value_n_iter_no_change'], model_cfg['min_value_n_iter_no_change']]
            activation = [model_cfg['max_value_activation'],
                          model_cfg['min_value_activation']]
            solver = [model_cfg['max_value_solver'],
                      model_cfg['min_value_solver']]
            learning_rate = [
                model_cfg['max_value_learning_rate'], model_cfg['min_value_learning_rate']]

            self.max_min = [hidden_layer, hidden_layer_2, alpha, learning_rate_init, max_iter,
                            tol, beta, beta_2, n_iter_no_change, activation, solver, learning_rate]
            if model_cfg["type"] == "classification":
                self.sklearn = MLPClassifier
            elif model_cfg["type"] == "regression":
                self.sklearn = MLPRegressor

        elif(model == "SVM"):
            c = [model_cfg['max_value_c'], model_cfg['min_value_c']]
            tol = [model_cfg['max_value_tol'], model_cfg['min_value_tol']]
            max_iter = [model_cfg['max_value_max_iter'],
                        model_cfg['min_value_max_iter']]
            gamma = [model_cfg['max_value_gamma'],
                     model_cfg['min_value_gamma']]
            kernal = [model_cfg['max_value_kernal'],
                      model_cfg['min_value_kernal']]
            self.max_min = [c, tol, max_iter, gamma, kernal]
            if model_cfg["type"] == "classification":
                self.sklearn = SVC
            elif model_cfg["type"] == "regression":
                self.sklearn = SVR

# initialize
    def virus_origin(self):
        current_parameter = [
            [0 for _ in range(self.virus_dim+3)]for _ in range(self.virus_num)]
        for i in range(self.virus_num):
            for j in range(self.virus_dim):
                random_number = random.random()
                if j in self.int_parameter:  # int limit
                    current_parameter[i][j] = int(
                        random_number * (self.max_min[j][0] - self.max_min[j][1]) + self.max_min[j][1])
                else:
                    current_parameter[i][j] = random_number * \
                        (self.max_min[j][0] - self.max_min[j]
                         [1]) + self.max_min[j][1]
        for i in range(self.virus_num):
            current_parameter[i][self.virus_dim+2] = i
        # label survived virus
        for i in range(len(current_parameter)):
            current_parameter[i][self.virus_dim] = 0
        return current_parameter

# calculate fitness
    def fitness(self, current_parameter, parms, train, test, record):
        for i in range(len(current_parameter)):
            if current_parameter[i][self.virus_dim] == 1:
                continue
            elif len(parms) == self.total_virus_limit:
                return current_parameter, parms, train, test, record
            elif current_parameter[i][self.virus_dim] == 0:
                if(self.model == "KNN"):
                    if(0 <= current_parameter[i][2] <= 4):
                        weights = "uniform"
                    elif(4 < current_parameter[i][2] <= 8):
                        weights = "distance"

                    if(0 <= current_parameter[i][3] <= 4):
                        algorithm = "ball_tree"
                    elif(4 < current_parameter[i][3] <= 8):
                        algorithm = "kd_tree"
                    elif(8 < current_parameter[i][3] <= 12):
                        algorithm = "brute"
                    elif(12 < current_parameter[i][3] <= 16):
                        algorithm = "auto"

                    if(0 <= current_parameter[i][4] <= 4):
                        metric = "euclidean"
                    elif(4 < current_parameter[i][4] <= 8):
                        metric = "manhattan"
                    elif(8 < current_parameter[i][4] <= 12):
                        metric = "chebyshev"

                    with parallel_backend('threading'):
                        knn = KNeighborsClassifier(
                            n_neighbors=current_parameter[i][0], weights=weights, algorithm=algorithm, leaf_size=current_parameter[i][1], metric=metric, n_jobs=-1)
                        cv_scores = cross_validate(
                            knn, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                    print(i, cv_scores['test_score'].mean())
                    if(record == True):
                        parms.append([current_parameter[i][0], current_parameter[i][1], weights,
                                      algorithm, metric]+current_parameter[i][self.virus_dim+2:])  # record

                elif(self.model == "MLP"):
                    if(0 <= current_parameter[i][9] <= 4):
                        activation = "identity"
                    elif(4 < current_parameter[i][9] <= 8):
                        activation = "logistic"
                    elif(8 < current_parameter[i][9] <= 12):
                        activation = "tanh"
                    elif(12 < current_parameter[i][9] <= 16):
                        activation = "relu"

                    if(0 <= current_parameter[i][10] <= 4):
                        solver = "lbfgs"
                    elif(4 < current_parameter[i][10] <= 8):
                        solver = "sgd"
                    elif(8 < current_parameter[i][10] <= 12):
                        solver = "adam"

                    if(0 <= current_parameter[i][11] <= 4):
                        learning_rate = "constant"
                    elif(4 < current_parameter[i][11] <= 8):
                        learning_rate = "invscaling"
                    elif(8 < current_parameter[i][11] <= 12):
                        learning_rate = "adaptive"

                    with parallel_backend('threading'):
                        mlp = MLPClassifier(hidden_layer_sizes=[current_parameter[i][0], current_parameter[i][1]], activation=activation, solver=solver, alpha=current_parameter[i][2], batch_size='auto',
                                            learning_rate=learning_rate, learning_rate_init=current_parameter[i][3], max_iter=current_parameter[i][4], shuffle=True, random_state=None, tol=current_parameter[i][5],
                                            verbose=False, warm_start=False, nesterovs_momentum=True, early_stopping=False, beta_1=current_parameter[i][6], beta_2=current_parameter[i][7], epsilon=1e-08,
                                            n_iter_no_change=current_parameter[i][8])
                        cv_scores = cross_validate(
                            mlp, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                    print(len(parms), cv_scores['test_score'].mean())
                    if(record == True):
                        parms.append(
                            [current_parameter[i][0], current_parameter[i][1],  current_parameter[i][2], current_parameter[i][3], current_parameter[i][4],
                             current_parameter[i][5], current_parameter[i][6], current_parameter[i][7], current_parameter[i][8], activation, solver, learning_rate] +
                            current_parameter[i][self.virus_dim+2:])

                elif(self.model == "SVM"):
                    if(0 <= current_parameter[i][4] <= 4):
                        kernel = "linear"
                    elif(4 < current_parameter[i][4] <= 8):
                        kernel = "poly"
                    elif(8 < current_parameter[i][4] <= 12):
                        kernel = "rbf"
                    elif(12 < current_parameter[i][4] <= 16):
                        kernel = "sigmoid"

                    with parallel_backend('threading'):
                        svm = SVC(C=current_parameter[i][0], kernel=kernel, degree=3, gamma=current_parameter[i][3], coef0=0.0, shrinking=True,
                                  probability=False, tol=current_parameter[i][1], cache_size=1000, class_weight=None, max_iter=current_parameter[i][2])
                        cv_scores = cross_validate(
                            svm, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                    print(len(parms), cv_scores['test_score'].mean())
                    if(record == True):
                        parms.append(
                            [current_parameter[i][0], current_parameter[i][1], current_parameter[i][2], current_parameter[i][3], kernel] +
                            current_parameter[i][self.virus_dim+2:])

                if np.isnan(cv_scores['test_score'].mean()):
                    current_parameter[i][self.virus_dim+1] = 0
                else:
                    current_parameter[i][self.virus_dim +
                                         1] = cv_scores['test_score'].mean()

                if(record == True):
                    test.append(cv_scores['test_score'].mean())  # test data
                    train.append(cv_scores['train_score'].mean())  # train data

        if(record == False):
            record = True
        return current_parameter, parms, train, test, record

# divided strong and common virus
    def classification(self, current_parameter):
        svn = []
        cvn = []
        current_parameter = sorted(
            current_parameter, key=lambda current_parameter: current_parameter[self.virus_dim+1], reverse=True)
        count_strong_virus = int(len(current_parameter)*self.s_proportion)
        if(count_strong_virus == 0):
            count_strong_virus = 1
        for i in range(count_strong_virus):  # divided according to ratio
            svn.append(current_parameter[i])
        for i in range(count_strong_virus, len(current_parameter)):
            cvn.append(current_parameter[i])
        return svn, cvn

    def record_best_virus(self, best_fitness, best_parameter, lb_fitness, lb_parameter, current_parameter):
        current_parameter = sorted(
            current_parameter, key=lambda current_parameter: current_parameter[self.virus_dim+1], reverse=True)
        if current_parameter[0][self.virus_dim+1] >= best_fitness:
            best_fitness = current_parameter[0][self.virus_dim+1]
            best_parameter = current_parameter[0][0:self.virus_dim]
        else:
            self.intensity = self.intensity+1
        lb_fitness.append(current_parameter[0][self.virus_dim+1])
        lb_parameter.append(current_parameter[0][0:self.virus_dim])
        return best_fitness, best_parameter, lb_fitness, lb_parameter

# clone new virus
    def update(self, current_parameter, svn, cvn, parms):
        svn_update = []
        cvn_update = []
        # clone strong virus
        for i in range(len(svn)):
            for k in range(self.strong_growth_rate):
                random_number = random.random()
                tmp = []
                for j in range(self.virus_dim+3):
                    if j == self.virus_dim+1:
                        tmp.append(0)
                    # label survived virus
                    elif j == self.virus_dim:
                        tmp.append(0)
                    elif j == self.virus_dim+2:
                        if (len(parms) == 0):
                            tmp.append(svn[i][self.virus_dim+2])
                            self.count = self.count+1
                            tmp.append(f"{self.count}(s)")
                        else:
                            for m in range(self.virus_dim+2, len(svn[i])):
                                tmp.append(svn[i][m])
                            self.count = self.count+1
                            tmp.append(f"{self.count}(s)")

                    else:
                        current_virus_parameter = svn[i][j]
                        # prevent to stop in maximum value
                        if (current_virus_parameter == self.max_min[j][0]):
                            m = (current_virus_parameter-(random_number /
                                                          self.intensity) * current_virus_parameter)
                            if (m < self.max_min[j][1]):
                                m = self.max_min[j][1]
                        # prevent to stop in minimum value
                        elif (current_virus_parameter == self.max_min[j][1]):
                            m = (current_virus_parameter+(random_number /
                                                          self.intensity) * current_virus_parameter)
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]
                        else:
                            m = current_virus_parameter + \
                                (random.uniform(-1, 1)/self.intensity) * \
                                current_virus_parameter
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]
                            elif (m < self.max_min[j][1]):
                                m = self.max_min[j][1]
                        if j in self.int_parameter:
                            tmp.append(int(m))
                        else:
                            tmp.append(m)

                svn_update.append(tmp)
        # clone common virus
        for i in range(len(cvn)):
            for k in range(self.common_growth_rate):
                random_number = random.random()
                tmp = []
                for j in range(self.virus_dim+3):
                    if j == self.virus_dim+1:
                        tmp.append(0)
                    # label survived virus
                    elif j == self.virus_dim:
                        tmp.append(0)
                    elif j == self.virus_dim+2:
                        if (len(parms) == 0):
                            tmp.append(cvn[i][self.virus_dim+2])
                            self.count = self.count+1
                            tmp.append(self.count)
                        else:
                            for m in range(self.virus_dim+2, len(cvn[i])):
                                tmp.append(cvn[i][m])
                            self.count = self.count+1
                            tmp.append(self.count)
                    else:
                        current_virus_parameter = cvn[i][j]

                        if (current_virus_parameter == self.max_min[j][0]):
                            m = (current_virus_parameter -
                                 random_number * current_virus_parameter)
                            if (m < self.max_min[j][1]):
                                m = self.max_min[j][1]

                        elif (current_virus_parameter == self.max_min[j][1]):
                            m = (current_virus_parameter +
                                 random_number * current_virus_parameter)
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]
                        else:
                            m = current_virus_parameter + \
                                random.uniform(-1, 1) * current_virus_parameter
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]
                            elif (m < self.max_min[j][1]):
                                m = self.max_min[j][1]
                        if j in self.int_parameter:
                            tmp.append(int(m))
                        else:
                            tmp.append(m)
                cvn_update.append(tmp)
        for i in range(len(svn_update)):  # merge all the virus
            current_parameter.append(svn_update[i])
        for i in range(len(cvn_update)):
            current_parameter.append(cvn_update[i])
        return current_parameter

# kill virus

    def Anti_Virus(self, current_parameter):
        count_strong_virus = int(len(current_parameter)*self.s_proportion)
        if(count_strong_virus == 0):
            count_strong_virus = 1
        count_common_virus = len(current_parameter)-count_strong_virus
        common_kill_amount = random.randint(
            int(count_common_virus*0.5), count_common_virus)  # generate random number of kills common virus
        strong_kill_amount = random.randint(
            0, int(count_strong_virus*0.5))  # generate random number of kills strong virus
        current_parameter = sorted(
            current_parameter, key=lambda current_parameter: current_parameter[self.virus_dim+1], reverse=True)
        # kill common virus
        for j in range(common_kill_amount):
            kill = random.sample(
                range(count_strong_virus, count_strong_virus+count_common_virus-j), 1)
            del current_parameter[kill[0]]
        # kill strong virus
        for j in range(strong_kill_amount):
            kill = random.sample(
                range(0, count_strong_virus-j-1), 1)
            logging.debug(f"killed virus:{current_parameter[kill[0]][-1]}")
            del current_parameter[kill[0]]
        # label survived virus

        for i in range(len(current_parameter)):
            current_parameter[i][self.virus_dim] = 1
        return current_parameter, common_kill_amount, strong_kill_amount

# 2.7 主函数
    def main(self):
        '''主函数
        '''
        parms = []
        train = []
        test = []
        best_fitness = 0
        best_parameter = []
        lb_fitness = []
        lb_parameter = []
        check_virus_limit = self.virus_num
        record = False
        iteration = 0
        # initialize
        current_parameter = self.virus_origin()
        current_parameter, parms, train, test, record = self.fitness(
            current_parameter, parms, train, test, record)

        print("VOA start!!")
        while(check_virus_limit <= self.total_virus_limit):
            svn, cvn = self.classification(current_parameter)
            current_parameter = self.update(
                current_parameter, svn, cvn, parms)

            current_parameter, parms, train, test, record = self.fitness(
                current_parameter, parms, train, test, record)

            best_fitness, best_parameter, lb_fitness, lb_parameter = self.record_best_virus(
                best_fitness, best_parameter, lb_fitness, lb_parameter, current_parameter)
            if len(parms) >= self.total_virus_limit:
                iteration = iteration+1
                log = f"iter: {iteration},local_best_fitness: {lb_fitness[iteration-1]},global best fitness: {best_fitness},totall_virus_count: {len(parms)},\
virus_count: {len(current_parameter)} VOA finished!!"
                logging.info(log)
                break
            # recored survived_virus_count
            survived_virus_count = pd.DataFrame(current_parameter)
            survived_virus_count = len(
                survived_virus_count[survived_virus_count[self.virus_dim] == 1])

            current_parameter, common_kill_amount, strong_kill_amount = self.Anti_Virus(
                current_parameter)

            iteration = iteration+1
            log = f"iter: {iteration},local_best_fitness: {lb_fitness[iteration-1]},global best fitness: {best_fitness},totall_virus_count: {len(parms)},\
virus_count_before_killing: {len(current_parameter)+strong_kill_amount+common_kill_amount},virus_count: {len(current_parameter)},Strong_kill: {strong_kill_amount},Common_kill: {common_kill_amount} survived virus: {survived_virus_count}"
            logging.info(log)
            check_virus_limit = len(parms)

        return best_fitness, best_parameter, lb_fitness, lb_parameter, parms, train, test
