import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_validate
from sklearn.utils import parallel_backend
import logging
from sklearn.metrics import make_scorer, recall_score
specificity = make_scorer(recall_score, pos_label=0)


class VOA(object):
    def __init__(self, virus_num, virus_dim, strong_growth_rate, common_growth_rate, s_proportion, total_virus_limit,
                 intensity, model_cfg, X, y, model, scoring, task_type, folder, file, algo_MLconfig, k_fold):
        self.k_fold = k_fold
        self.virus_num = virus_num
        self.virus_dim = virus_dim
        self.s_proportion = s_proportion
        self.strong_growth_rate = strong_growth_rate
        self.common_growth_rate = common_growth_rate
        self.total_virus_limit = total_virus_limit
        self.intensity = intensity

        self.int_param = []
        self.class_param = []
        self.param_name = []
        for index, i in enumerate(model_cfg.keys()):
            if model_cfg[i]['int']:
                self.int_param.append(index)
            if model_cfg[i]['class']:
                self.class_param.append(index)
                self.param_name.append(model_cfg[i]["class_name"])
            else:
                self.param_name.append([])
        self.keys = model_cfg.keys()
        self.count = self.virus_num-1
        self.X = X
        self.y = y
        self.model = model
        if scoring == 'specificity':
            self.scoring = specificity
        else:
            self.scoring = scoring
        self.folder = folder
        self.file = file
        # -------------------------------------------------------------
        self.max_min, self.ML_model = algo_MLconfig(
            model, task_type, model_cfg)

        # initialize

    def virus_origin(self):
        current_parameter = [
            [0 for _ in range(self.virus_dim+3)]for _ in range(self.virus_num)]
        for i in range(self.virus_num):
            for j in range(self.virus_dim):
                random_number = random.random()
                if j in self.int_param:  # int limit
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
    def fitness(self, current_parameter, params, train, test, record):
        for i in range(len(current_parameter)):
            if current_parameter[i][self.virus_dim] == 1:
                continue
            elif len(params) == self.total_virus_limit:
                return current_parameter, params, train, test, record
            elif current_parameter[i][self.virus_dim] == 0:
                class_particle = []
                for index, class_num in enumerate(self.class_param):
                    count = 1
                    for class_name in self.param_name[class_num]:
                        if(count-1 < current_parameter[i][class_num] <= count):
                            class_particle.append(class_name)
                            break
                        else:
                            count += 1
                if(self.model == "KNN"):
                    with parallel_backend('threading'):
                        knn = self.ML_model(
                            n_neighbors=current_parameter[i][0], weights=class_particle[0], algorithm=class_particle[1], leaf_size=current_parameter[i][1], metric=class_particle[2], n_jobs=-1)
                        cv_scores = cross_validate(
                            knn, self.X, self.y, cv=self.k_fold, scoring=self.scoring, n_jobs=-1, return_train_score=True)
                    print(len(params), cv_scores['test_score'].mean())
                    if(record == True):
                        params.append([current_parameter[i][0], current_parameter[i][1], class_particle[0],
                                       class_particle[1], class_particle[2]]+current_parameter[i][self.virus_dim+2:])  # record
                elif(self.model == "ADA"):
                    with parallel_backend('threading'):
                        ada = self.ML_model[0](n_estimators=current_parameter[i][0], learning_rate=current_parameter[i][1], algorithm=class_particle[0], base_estimator=self.ML_model[1](
                            criterion=class_particle[1], max_depth=current_parameter[i][4], min_samples_split=current_parameter[i][5], min_samples_leaf=current_parameter[i][6], max_features=class_particle[2]))
                        cv_scores = cross_validate(
                            ada, self.X, self.y, cv=self.k_fold, scoring=self.scoring, n_jobs=-1, return_train_score=True)
                    print(len(params), cv_scores['test_score'].mean())
                    if(record == True):
                        params.append([current_parameter[i][0], current_parameter[i][1], class_particle[0],
                                       class_particle[1], current_parameter[i][4], current_parameter[i][5], current_parameter[i][6], class_particle[2]])
                elif(self.model == "RF"):
                    with parallel_backend('threading'):
                        rf = self.ML_model(n_estimators=current_parameter[i][0], criterion=class_particle[0],
                                           max_depth=current_parameter[i][2], min_samples_split=current_parameter[
                                               i][3], min_samples_leaf=current_parameter[i][4],
                                           max_features=class_particle[1], n_jobs=-1)
                        cv_scores = cross_validate(
                            rf, self.X, self.y, cv=self.k_fold, scoring=self.scoring, n_jobs=-1, return_train_score=True)
                    print(len(params), cv_scores['test_score'].mean())
                    if(record == True):
                        params.append([current_parameter[i][0], class_particle[0], current_parameter[i]
                                       [2], current_parameter[i][3], current_parameter[i][4], class_particle[1]])

                elif(self.model == "MLP"):
                    with parallel_backend('threading'):
                        mlp = self.ML_model(hidden_layer_sizes=[current_parameter[i][0], current_parameter[i][1]], activation=class_particle[0], solver=class_particle[1], alpha=current_parameter[i][2],
                                            learning_rate=class_particle[2], learning_rate_init=current_parameter[i][
                                                3], max_iter=current_parameter[i][4],  tol=current_parameter[i][5],
                                            beta_1=current_parameter[i][6], beta_2=current_parameter[i][7],
                                            n_iter_no_change=current_parameter[i][8])
                        cv_scores = cross_validate(
                            mlp, self.X, self.y, cv=self.k_fold, scoring=self.scoring, n_jobs=-1, return_train_score=True)
                    print(len(params), cv_scores['test_score'].mean())
                    if(record == True):
                        params.append(
                            [current_parameter[i][0], current_parameter[i][1],  current_parameter[i][2], current_parameter[i][3], current_parameter[i][4],
                             current_parameter[i][5], current_parameter[i][6], current_parameter[i][7], current_parameter[i][8], class_particle[0], class_particle[1], class_particle[2]] +
                            current_parameter[i][self.virus_dim+2:])

                elif(self.model == "SVM"):
                    with parallel_backend('threading'):
                        svm = self.ML_model(C=current_parameter[i][0], kernel=class_particle[0],  gamma=current_parameter[i]
                                            [3], tol=current_parameter[i][1], cache_size=1000, max_iter=current_parameter[i][2])
                        cv_scores = cross_validate(
                            svm, self.X, self.y, cv=self.k_fold, scoring=self.scoring, n_jobs=-1, return_train_score=True)
                    print(len(params), cv_scores['test_score'].mean())
                    if(record == True):
                        params.append(
                            [current_parameter[i][0], current_parameter[i][1], current_parameter[i][2], current_parameter[i][3], class_particle[0]] +
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
        return current_parameter, params, train, test, record

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
    def update(self, current_parameter, svn, cvn, params):
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
                        if (len(params) == 0):
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
                        if j in self.int_param:
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
                        if (len(params) == 0):
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
                        if j in self.int_param:
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
            # logging.debug(
            #   f"killed common virus:{current_parameter[kill[0]][-1]}")
            del current_parameter[kill[0]]
        # kill strong virus
        for j in range(strong_kill_amount):
            kill = random.sample(
                range(0, count_strong_virus-j-1), 1)
            logging.debug(
                f"killed strong virus:{current_parameter[kill[0]][-1]}")
            del current_parameter[kill[0]]
        # label survived virus

        for i in range(len(current_parameter)):
            current_parameter[i][self.virus_dim] = 1
        return current_parameter, common_kill_amount, strong_kill_amount

    def plot_curve(self, lb_fitness):
        plt.figure(figsize=(10, 6))
        plt.title("VOA Algorithm", fontsize=20)
        plt.xlabel("iterations", fontsize=15, labelpad=15)
        plt.ylabel("f(gb)", fontsize=15, labelpad=20)
        plt.plot(lb_fitness, color='b')
        plt.savefig(
            f'{self.folder}/VOA_{self.model}_{self.file}')

    def main(self):
        params = []
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
        current_parameter, params, train, test, record = self.fitness(
            current_parameter, params, train, test, record)

        print("VOA start!!")
        while(check_virus_limit <= self.total_virus_limit):
            svn, cvn = self.classification(current_parameter)
            current_parameter = self.update(
                current_parameter, svn, cvn, params)

            current_parameter, params, train, test, record = self.fitness(
                current_parameter, params, train, test, record)

            best_fitness, best_parameter, lb_fitness, lb_parameter = self.record_best_virus(
                best_fitness, best_parameter, lb_fitness, lb_parameter, current_parameter)
            if len(params) >= self.total_virus_limit:
                iteration = iteration+1
                log = f"iter: {iteration},local_best_fitness: {lb_fitness[iteration-1]},global best fitness: {best_fitness},totall_virus_count: {len(params)},\
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
            log = f"iter: {iteration},local_best_fitness: {lb_fitness[iteration-1]},global best fitness: {best_fitness},totall_virus_count: {len(params)},\
virus_count_before_killing: {len(current_parameter)+strong_kill_amount+common_kill_amount},virus_count: {len(current_parameter)},Strong_kill: {strong_kill_amount},Common_kill: {common_kill_amount} survived virus: {survived_virus_count}"
            logging.info(log)
            check_virus_limit = len(params)

        self.plot_curve(lb_fitness)
        logging.info(
            (f"best_parameter:{best_parameter},best_fitness:{best_fitness}"))

        params_number = [i for i in range(self.virus_dim)]

        data_params = pd.DataFrame(params)[params_number]
        data_params.columns = self.keys
        updata_history = pd.DataFrame(params).drop(params_number, axis=1)
        results = pd.concat(
            [pd.DataFrame(train, columns=["train"]), pd.DataFrame(test, columns=["test"]), data_params, updata_history], axis=1)

        return results, self.ML_model, best_parameter, self.class_param, self.param_name
