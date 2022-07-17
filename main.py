import datetime
import logging
import json
import argparse
from funcitons import *
import datetime
import os
from feature_selection import *

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
for _ in logging.root.manager.loggerDict:
    logging.getLogger(_).setLevel(logging.CRITICAL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="KNN,MLP,SVM,RF,ADA", default="ADA")
    parser.add_argument(
        "--data", help="custom", default="custom")
    parser.add_argument("--algo", help="PSO,VOA", default="PSO")
    parser.add_argument(
        "--scoring", help="cls: accuracy, f1, recall, precision, specificity; reg: r2, neg_mean_absolute_error, neg_mean_squared_error", default="accuracy")
    parser.add_argument(
        "--k_fold", help="set k value , need to >1", default=5, type=int)
    parser.add_argument(
        "--confusion_m", help="Do you need to gernerate the confusion_matrix?(False or True)", default=False, type=bool)
    parser.add_argument(
        "--feat_select", help="feature selection", default=False, type=bool)
    parser.add_argument(
        "--feat_select_type", help="chi2, pearson, ANOVA, MIC ", default="chi2", type=str)

    args = parser.parse_args()

    with open('./dataset/custom.json') as f:
        custom_data = json.load(f)
    dataset, task_type, y, X, file, target, data_path = load_data(
        args.data, custom_data)

    if(args.feat_select == True):
        feat_score(args.feat_select_type, task_type, y, X, file)

    else:
        model_cfg = load_ML_model_cfg(args)

        rightnow = str(datetime.datetime.today()).replace(
            " ", "_").replace(":", "-")[:-7]
        folder = f'./results/{rightnow}_{file}_{args.model}_{args.algo}'
        os.makedirs(folder)
        log_path = f'{folder}/{args.algo}({args.model})_{args.data}.log'
        logging.basicConfig(
            level=logging.DEBUG, filename=log_path, filemode='w', format=FORMAT)

        print(
            f'Algo: {args.algo}, Model: {args.model}, Data: {args.data}, scoring: {args.scoring}, k_fold: {args.k_fold}, confusion_m: {args.confusion_m}')
        logging.info(
            f'Algo: {args.algo}, Model: {args.model}, Data: {args.data}, scoring: {args.scoring}, k_fold: {args.k_fold}, confusion_m: {args.confusion_m}')

        print(f'data_path = {data_path}')
        logging.info(f'data_path = {data_path}')

        print(f'data:{len(dataset)}, y={target} , X_num = {len(X.columns)}')
        logging.info(
            f'data:{len(dataset)}, y={target} , X_num = {len(X.columns)}')

        print(f'{args.model} config: {model_cfg}')
        logging.info(f'{args.model} config: {model_cfg}')

        hyper(args, model_cfg, X, y, file, folder, task_type)
