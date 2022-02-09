
import pandas as pd
import numpy as np
import random
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.utils import parallel_backend
from random import sample
import logging
from tqdm import tqdm, trange

'''from sklearn.metrics import make_scorer, recall_score
specificity = make_scorer(recall_score, pos_label=0)'''




class VOA(object):
    def __init__(self, virus_num, virus_dim, strong_growth_rate,common_growth_rate, s_proportion, total_virus_limit, 
    intensity, model_cfg,X,y,model,matrix):

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
        self.matrix = matrix
        #-------------------------------------------------------------
        if(model == "KNN"):
            #knn參數最大值跟最小值設定
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
            n_iter_no_change = [model_cfg['max_value_n_iter_no_change'], model_cfg['min_value_n_iter_no_change']]
            activation = [model_cfg['max_value_activation'],model_cfg['min_value_activation']]
            solver = [model_cfg['max_value_solver'],
                           model_cfg['min_value_solver']]
            learning_rate = [
                model_cfg['max_value_learning_rate'], model_cfg['min_value_learning_rate']]

            self.max_min = [hidden_layer, hidden_layer_2, alpha,learning_rate_init, max_iter, tol, beta, beta_2, n_iter_no_change, activation, solver, learning_rate]

        elif(model == "SVM"):
            c = [model_cfg['max_value_c'], model_cfg['min_value_c']]
            tol = [model_cfg['max_value_tol'],model_cfg['min_value_tol']]
            max_iter = [model_cfg['max_value_max_iter'],
                             model_cfg['min_value_max_iter']]
            gamma = [model_cfg['max_value_gamma'],
                          model_cfg['min_value_gamma']]
            kernal = [model_cfg['max_value_kernal'],
                           model_cfg['min_value_kernal']]
            self.max_min = [c, tol, max_iter, gamma, kernal]

### 2.1 宿主细胞病毒初始化
    def virus_origin(self):
        current_parameter = [
            [0 for _ in range(self.virus_dim+3)]for _ in range(self.virus_num)]
        for i in range(self.virus_num):
            for j in range(self.virus_dim):
                random_number = random.random()
                if j in self.int_parameter:  # 判斷 如果超參數有整數限制
                    current_parameter[i][j] = int(
                        random_number * (self.max_min[j][0] - self.max_min[j][1]) + self.max_min[j][1])
                else:
                    current_parameter[i][j] = random_number * \
                        (self.max_min[j][0] - self.max_min[j]
                         [1]) + self.max_min[j][1]
        for i in range(self.virus_num):
            current_parameter[i][self.virus_dim+2] = i
        #label survived virus
        for i in range(len(current_parameter)):
            current_parameter[i][self.virus_dim] = 0
        return current_parameter

## 2.2 计算适应度函数数值列表
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
                        parms.append([current_parameter[i][0], current_parameter[i][1], weights, algorithm, metric]+current_parameter[i][self.virus_dim+2:])  # 紀錄參數

                elif(self.model =="MLP"):
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
                            current_parameter[i][self.virus_dim+2:])  # 紀錄參數])

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
                            current_parameter[i][self.virus_dim+2:])  # 紀錄參數])
                '''cv_scores = pd.DataFrame(data=[[random.random(), random.random()], [random.random(), random.random()]], columns=[
                                         'test_score', 'train_score'])'''


                if np.isnan(cv_scores['test_score'].mean()):
                    current_parameter[i][self.virus_dim+1] = 0
                else:
                    current_parameter[i][self.virus_dim +1] = cv_scores['test_score'].mean()

                if(record == True):
                    test.append(cv_scores['test_score'].mean())  # 紀錄測試資料
                    train.append(cv_scores['train_score'].mean())  # 紀錄訓練資料

        if(record == False):
            record = True
        return current_parameter, parms, train, test, record

## 2.3 强弱病毒分類
    def classification(self, current_parameter):
        svn = []  # 儲存強壯病毒
        cvn = []  # 儲存一般病毒
        current_parameter = sorted(
            current_parameter, key=lambda current_parameter: current_parameter[self.virus_dim+1], reverse=True)
        count_strong_virus = int(len(current_parameter)*self.s_proportion)
        if(count_strong_virus == 0):
            count_strong_virus = 1
        for i in range(count_strong_virus):  # 按比例分病毒種類
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

## 2.4  复制新病毒
    def update(self, current_parameter, svn, cvn, parms):
        svn_update = []  # 強病毒新複製病毒
        cvn_update = []  # 弱病毒新複製病毒
        ## 1.复制强病毒（产生新的待寻优参数值）
        for i in range(len(svn)):
            for k in range(self.strong_growth_rate):
                random_number = random.random()
                tmp = []
                for j in range(self.virus_dim+3):
                    if j == self.virus_dim+1:
                        tmp.append(0)
                    #label survived virus
                    elif j == self.virus_dim:
                        tmp.append(0)
                    elif j == self.virus_dim+2:
                        if (len(parms) == 0):
                            tmp.append(svn[i][self.virus_dim+2])  # 存母强病毒编号
                            self.count = self.count+1
                            tmp.append(f"{self.count}(s)")  # 存入自己的编号
                        else:
                            for m in range(self.virus_dim+2, len(svn[i])):
                                tmp.append(svn[i][m])  # 存母躰病毒歷代信息
                            self.count = self.count+1
                            tmp.append(f"{self.count}(s)")  # 存入自己的编号

                    else:
                        current_virus_parameter = svn[i][j]
                        # 防止停留在最大值
                        if (current_virus_parameter == self.max_min[j][0]):
                            m = (current_virus_parameter-(random_number /
                                                          self.intensity) * current_virus_parameter)
                            if (m < self.max_min[j][1]):
                                m = self.max_min[j][1]  # 超最小值处理：取为最小值
                        # 防止停留在最小值
                        elif (current_virus_parameter == self.max_min[j][1]):
                            m = (current_virus_parameter+(random_number /
                                                          self.intensity) * current_virus_parameter)
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]  # 超最大值处理：取为最大值
                        else:
                            m = current_virus_parameter + \
                                (random.uniform(-1, 1)/self.intensity) * \
                                current_virus_parameter
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]  # 超最大值处理：取为最大值
                            elif (m < self.max_min[j][1]):
                                m = self.max_min[j][1]  # 超最小值处理：取为最小值
                        if j in self.int_parameter:
                            tmp.append(int(m))
                        else:
                            tmp.append(m)

                svn_update.append(tmp)
        ## 1.复制弱病毒（产生新的待寻优参数值）
        for i in range(len(cvn)):
            for k in range(self.common_growth_rate):
                random_number = random.random()
                tmp = []
                for j in range(self.virus_dim+3):
                    if j == self.virus_dim+1:
                        tmp.append(0)
                    #label survived virus
                    elif j == self.virus_dim:
                        tmp.append(0)
                    elif j == self.virus_dim+2:
                        if (len(parms) == 0):
                            tmp.append(cvn[i][self.virus_dim+2])  # 存母强病毒编号
                            self.count = self.count+1
                            tmp.append(self.count)  # 存入自己的编号
                        else:
                            for m in range(self.virus_dim+2, len(cvn[i])):
                                tmp.append(cvn[i][m])  # 存母躰病毒歷代信息
                            self.count = self.count+1
                            tmp.append(self.count)  # 存入自己的编号
                    else:
                        current_virus_parameter = cvn[i][j]
                        # 防止停留在最大值
                        if (current_virus_parameter == self.max_min[j][0]):
                            m = (current_virus_parameter -
                                 random_number * current_virus_parameter)
                            if (m < self.max_min[j][1]):
                                m = self.max_min[j][1]  # 超最小值处理：取为最小值
                        # 防止停留在最小值
                        elif (current_virus_parameter == self.max_min[j][1]):
                            m = (current_virus_parameter +
                                 random_number * current_virus_parameter)
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]  # 超最大值处理：取为最大值
                        else:
                            m = current_virus_parameter + \
                                random.uniform(-1, 1) * current_virus_parameter
                            if (m > self.max_min[j][0]):
                                m = self.max_min[j][0]  # 超最大值处理：取为最大值
                            elif (m < self.max_min[j][1]):
                                m = self.max_min[j][1]  # 超最小值处理：取为最小值
                        if j in self.int_parameter:
                            tmp.append(int(m))
                        else:
                            tmp.append(m)
                cvn_update.append(tmp)
        for i in range(len(svn_update)):  # 合併更新後及更新前病毒
            current_parameter.append(svn_update[i])
        for i in range(len(cvn_update)):
            current_parameter.append(cvn_update[i])
        return current_parameter

## 2.6 免疫杀毒

    def Anti_Virus(self, current_parameter):
        count_strong_virus = int(len(current_parameter)*self.s_proportion)
        if(count_strong_virus == 0):
            count_strong_virus = 1
        count_common_virus = len(current_parameter)-count_strong_virus
        common_kill_amount = random.randint(
            int(count_common_virus*0.5), count_common_virus)  # 隨機生成common殺毒數量
        strong_kill_amount = random.randint(
            0, int(count_strong_virus*0.5))  # 隨機生成strong殺毒數量
        current_parameter = sorted(current_parameter, key=lambda current_parameter: current_parameter[self.virus_dim+1], reverse=True)
        #kill common virus
        for j in range(common_kill_amount):
            kill = random.sample(
                range(count_strong_virus, count_strong_virus+count_common_virus-j), 1)
            del current_parameter[kill[0]]
        #kill strong virus
        for j in range(strong_kill_amount):
            kill = random.sample(
                range(0, count_strong_virus-j-1), 1)
            logging.debug(f"killed virus:{current_parameter[kill[0]][-1]}")
            del current_parameter[kill[0]]
        # label survived virus

        for i in range(len(current_parameter)):
            current_parameter[i][self.virus_dim] = 1
        return current_parameter, common_kill_amount, strong_kill_amount

## 2.7 主函数
    def main(self):
        '''主函数
        '''
        parms = []
        train = []
        test = []
        best_fitness = 0  # 紀錄最好的病毒表現
        best_parameter = []
        lb_fitness = []
        lb_parameter = []
        check_virus_limit = self.virus_num
        record = False
        iteration = 0
        ## 1、病毒群初始化
        current_parameter = self.virus_origin()
        ## 2、初始化目标函数值
        current_parameter, parms, train, test, record = self.fitness(
            current_parameter, parms, train, test, record)
        ## 4、迭代
        print("VOA start!!")
        while(check_virus_limit <= self.total_virus_limit):
            svn, cvn = self.classification(current_parameter)  # 分強弱病毒
            current_parameter = self.update(
                current_parameter, svn, cvn, parms)  # 病毒複製

            current_parameter, parms, train, test, record = self.fitness(
                current_parameter, parms, train, test, record)  # 進行機器學習

            best_fitness, best_parameter, lb_fitness, lb_parameter = self.record_best_virus(
                best_fitness, best_parameter, lb_fitness, lb_parameter, current_parameter)  # 紀錄最佳表現
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
                current_parameter)  # 殺毒

            iteration = iteration+1
            log = f"iter: {iteration},local_best_fitness: {lb_fitness[iteration-1]},global best fitness: {best_fitness},totall_virus_count: {len(parms)},\
virus_count_before_killing: {len(current_parameter)+strong_kill_amount+common_kill_amount},virus_count: {len(current_parameter)},Strong_kill: {strong_kill_amount},Common_kill: {common_kill_amount} survived virus: {survived_virus_count}"
            logging.info(log)
            check_virus_limit = len(parms)

        return best_fitness, best_parameter, lb_fitness, lb_parameter, parms, train, test
