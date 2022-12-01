import datetime
import logging
import json
import argparse
import datetime
import os
from task.hyper import *
from task.feature_selection import *
from task.clustering import *
from utility import *

FORMAT = '%(asctime)s %(levelname)s: %(message)s'
for _ in logging.root.manager.loggerDict:
    logging.getLogger(_).setLevel(logging.CRITICAL)

def _build_parser():
    parser = argparse.ArgumentParser()
    subcmd = parser.add_subparsers(dest='task', help='which task you want to run,[hyper] or [cluster] or [feat_select]', metavar='TASK')
    subcmd.required = True
    
    hyper_parser = subcmd.add_parser('hyper',help='hyperparameter algorithm')
    hyper_parser.add_argument("-m","--model", help="KNN,MLP,SVM,RF,ADA,XGBoost", default="KNN")
    hyper_parser.add_argument("-a","--algo", help="PSO,VOA,RANDOM", default="PSO")
    hyper_parser.add_argument("-s","--scoring", help="cls: accuracy, f1, recall, precision, specificity; reg: r2, neg_mean_absolute_error, neg_mean_squared_error", default="r2")
    hyper_parser.add_argument("-k","--k_fold", help="set k value , need to >1", default=5, type=int)
    hyper_parser.add_argument("-c","--confusion_m", help="Do you need to gernerate the confusion_matrix?(False or True)", default=False, type=bool)

    cluster_parser = subcmd.add_parser('cluster',help='clustering algorithm')
    cluster_parser.add_argument("-a","--algo", help="KMeans, DBSCAN", default='DBSCAN', type=str)

    featselect_parser = subcmd.add_parser('feat_select',help='feature selection algorithm')
    featselect_parser.add_argument("-m","--metric", help="chi2, pearson, ANOVA, MIC ", default="chi2", type=str)

    return parser

if __name__ == '__main__':
    parser = _build_parser()
    args = parser.parse_args()

    with open('./dataset/data_loader.json') as f:
        custom_data = json.load(f)
    dataset, task_type, y, X, file, target, data_path = load_data(custom_data)

    if(args.task == "feat_select"):
        #TODO create log
        feat_score(args.metric, task_type, y, X, file)

    elif(args.task == "cluster"):
        #TODO create log
        clustering(args.algo, y, X, file)

    elif(args.task == "hyper"):
        model_cfg = load_ML_model_cfg(args)

        rightnow = str(datetime.datetime.today()).replace(
            " ", "_").replace(":", "-")[:-7]
        folder = f'./results/{rightnow}_{file}_{args.model}_{args.algo}'
        os.makedirs(folder)
        log_path = f'{folder}/{args.algo}({args.model}).log'
        logging.basicConfig(
            level=logging.DEBUG, filename=log_path, filemode='w', format=FORMAT)

        print(
            f'Algo: {args.algo}, Model: {args.model}, scoring: {args.scoring}, k_fold: {args.k_fold}, confusion_m: {args.confusion_m}')
        logging.info(
            f'Algo: {args.algo}, Model: {args.model}, scoring: {args.scoring}, k_fold: {args.k_fold}, confusion_m: {args.confusion_m}')

        print(f'data_path = {data_path}')
        logging.info(f'data_path = {data_path}')

        print(f'data:{len(dataset)}, y={target} , X_num = {len(X.columns)}')
        logging.info(
            f'data:{len(dataset)}, y={target} , X_num = {len(X.columns)}')

        print(f'{args.model} config: {model_cfg}')
        logging.info(f'{args.model} config: {model_cfg}')

        hyper(args, model_cfg, X, y, file, folder, task_type)

    else:
        print(f"Unknown task: {args.task} \nwhich task you want to run,[hyper] or [clustering] or [feat_select]")