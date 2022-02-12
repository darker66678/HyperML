

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
from tqdm import tqdm
from sklearn.metrics import make_scorer, recall_score
specificity = make_scorer(recall_score, pos_label=0)


class PSO(object):
    def __init__(self, particle_num, particle_dim, iter_num, c1, c2, w, model_cfg, X, y, file, model, folder, matrix, task_type, algo_MLconfig):
        self.folder = folder
        self.model = model
        self.int_parameter = model_cfg['int_parameter']
        self.class_parm = model_cfg['class_parameter']
        self.param_name = model_cfg['param']
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
        self.max_min, self.ML_model = algo_MLconfig(
            model, task_type, model_cfg)

    def swarm_origin(self):
        particle_loc = []
        particle_dir = []
        for i in range(self.particle_num):
            tmp1 = []
            tmp2 = []
            for j in range(self.particle_dim):
                a = random.random()
                if j in self.int_parameter:
                    tmp1.append(
                        int(a * (self.max_min[j][0] - self.max_min[j][1]) + self.max_min[j][1]))
                else:
                    tmp1.append(
                        (a * (self.max_min[j][0] - self.max_min[j][1]) + self.max_min[j][1]))
                tmp2.append(
                    random.uniform(-self.max_min[j][0] / 3, self.max_min[j][0] / 3))
            particle_loc.append(tmp1)
            particle_dir.append(tmp2)
        return particle_loc, particle_dir

    def fitness(self, particle_loc, test, train, parms):
        fitness_value = []
        for i in range(self.particle_num):
            class_particle = []
            for index, class_num in enumerate(self.class_parm['class_number']):
                count = 1
                for class_name in self.class_parm['class_name'][index]:
                    if(count-1 < particle_loc[i][class_num] <= count):
                        class_particle.append(class_name)
                        break
                    else:
                        count += 1

            if(self.model == "KNN"):
                with parallel_backend('threading'):
                    knn = self.ML_model(n_neighbors=particle_loc[i][0], weights=class_particle[0], algorithm=class_particle[1],
                                        leaf_size=particle_loc[i][1], metric=class_particle[2], metric_params=None, n_jobs=-1,)
                    cv_scores = cross_validate(
                        knn, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                parms.append([particle_loc[i][0], particle_loc[i]
                              [1], class_particle[0], class_particle[1], class_particle[2]])

            elif(self.model == "MLP"):
                with parallel_backend('threading'):
                    mlp = self.ML_model(hidden_layer_sizes=[particle_loc[i][0], particle_loc[i][1]], activation=class_particle[0], solver=class_particle[1], alpha=particle_loc[i][2], batch_size='auto', learning_rate=class_particle[2], learning_rate_init=particle_loc[i][3], max_iter=particle_loc[i]
                                        [4], shuffle=True, random_state=None, tol=particle_loc[i][5], verbose=False, warm_start=False, nesterovs_momentum=True, early_stopping=False, beta_1=particle_loc[i][6], beta_2=particle_loc[i][7], epsilon=1e-08, n_iter_no_change=particle_loc[i][8])
                    cv_scores = cross_validate(
                        mlp, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                parms.append([particle_loc[i][0], particle_loc[i][1], particle_loc[i][2], particle_loc[i][3], particle_loc[i][4],
                              particle_loc[i][5], particle_loc[i][6], particle_loc[i][7], particle_loc[i][8], class_particle[0], class_particle[1], class_particle[2]])

            elif (self.model == "SVM"):
                with parallel_backend('threading'):
                    svm = self.ML_model(C=particle_loc[i][0], tol=particle_loc[i][1],
                                        max_iter=particle_loc[i][2], gamma=particle_loc[i][3], kernel=class_particle[0], cache_size=1000)
                    cv_scores = cross_validate(
                        svm, self.X, self.y, cv=5, scoring=self.matrix, n_jobs=-1, return_train_score=True)
                parms.append([particle_loc[i][0], particle_loc[i][1],
                              particle_loc[i][2], particle_loc[i][3], class_particle[0]])

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
                if j in self.int_parameter:
                    particle_loc[i][j] = int(((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                        self.max_min[j][0] - self.max_min[j][1]) + self.max_min[j][1]))
                else:
                    particle_loc[i][j] = (((particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (
                        self.max_min[j][0] - self.max_min[j][1]) + self.max_min[j][1]))

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

        logging.info(
            f'Final parameters are: , {self.param_name}, \n, {gbest_parameter}')
        print('Final parameters are :', self.param_name, "\n", gbest_parameter)
        results = pd.concat(
            [pd.Series(train), pd.Series(test)], axis=1)
        results = pd.concat(
            [pd.DataFrame(results), pd.DataFrame(parms)], axis=1)
        results.columns = ['train', 'test'] + self.param_name
        return results, self.ML_model, gbest_parameter, self.class_parm
