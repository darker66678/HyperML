import datetime
import logging
import json
import argparse
from funcitons import *
import datetime
import os


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.getLogger('matplotlib.font_manager').disabled = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="KNN,MLP,SVM", default="KNN")
    parser.add_argument(
        "--data", help="custom", default="custom")
    parser.add_argument("--algo", help="PSO,VOA", default="VOA")
    parser.add_argument(
        "--matrix", help="cls: accuracy, f1, recall, precision, specificity; reg: r2, neg_mean_absolute_error, neg_mean_squared_error", default="accuracy")
    args = parser.parse_args()

    with open('./dataset/custom.json') as f:
        custom_data = json.load(f)
    dataset, task_type, y, X, file, target, data_path = load_data(
        args.data, custom_data)

    model_cfg, model_cfg_json = load_ML_model_cfg(args)

    rightnow = str(datetime.datetime.today()).replace(
        " ", "_").replace(":", "-")[:-7]
    folder = f'./results/{rightnow}_{file}_{args.model}_{args.algo}'
    os.makedirs(folder)
    log_path = f'{folder}/{args.algo}({args.model})_{args.data}.log'
    logging.basicConfig(
        level=logging.DEBUG, filename=log_path, filemode='w', format=FORMAT)

    print(
        f'Algo: {args.algo}, Model: {args.model}, Data: {args.data}, Matrix: {args.matrix}')
    logging.info(
        f'Algo: {args.algo}, Model: {args.model}, Data: {args.data}, Matrix: {args.matrix}')

    print(f'data_path = {data_path}')
    logging.info(f'data_path = {data_path}')

    print(f'data:{len(dataset)}, y={target} , X_num = {len(X.columns)}')
    logging.info(
        f'data:{len(dataset)}, y={target} , X_num = {len(X.columns)}')

    print(f'{args.model} config: {model_cfg_json}')
    logging.info(f'{args.model} config: {model_cfg_json}')

    hyper(args, model_cfg, X, y, file, folder, task_type)
