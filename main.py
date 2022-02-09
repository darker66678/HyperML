import datetime
import logging
from PSO import PSO
from VOA import *
import json
import argparse
from funcitons import *
import datetime
import os


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.getLogger('matplotlib.font_manager').disabled = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",help="KNN,MLP,SVM",default="KNN")
    parser.add_argument(
        "--data", help="cancer,machine,custom", default="custom")
    parser.add_argument("--algo", help="PSO,VOA", default="PSO")
    parser.add_argument(
        "--matrix", help="accuracy, f1, recall, precision", default="accuracy")
    args = parser.parse_args()



    if(args.data == "custom"):
        with open('./dataset/custom.json') as f:
            custom_data = json.load(f)
        dataset = load_data(args.data, custom_data)
    else:
        dataset = load_data(args.data)

    if(args.model == "KNN"):
        with open('./cfg/ml_model/KNN_config.json') as f:
            model_cfg_json = json.load(f)
        model_cfg = load_KNN_cfg(model_cfg_json)

    elif(args.model == "MLP"):
        with open('./cfg/ml_model/MLP_config.json') as f:
            model_cfg_json = json.load(f)
        model_cfg = load_MLP_cfg(model_cfg_json)

    elif(args.model == "SVM"):
        with open('./cfg/ml_model/SVM_config.json') as f:
            model_cfg_json = json.load(f)
        model_cfg = load_SVM_cfg(model_cfg_json)

    for i, data in enumerate(dataset):
        if(args.data == "cancer"):
            if(i == 0):
                y = data.iloc[:, 48]
                X = data.iloc[:, 0:48]
                file = "ovarian"
            elif(i == 1):
                y = data.iloc[:, 9]
                X = data.iloc[:, 0:9]
                file = "breast"
            elif(i == 2):
                y = data.iloc[:, 18]
                X = data.iloc[:, 0:18]
                file = "cervical"

        elif (i == 0 and args.data == "machine"):
            y = data.iloc[:, 17]
            X = data.iloc[:, 0:17]
            file = "machine"

        elif (i == 0 and args.data == "custom"):
            y = data[custom_data['target']]
            X = data.drop([custom_data['target']],axis=1)
            target = custom_data['target']
            data_path = custom_data['data_path']
            file = custom_data['data_name']

        rightnow = str(datetime.datetime.today()).replace(
            " ", "_").replace(":", "-")[:-7]
        folder = f'./results/{rightnow}_{file}_{args.model}_{args.algo}'
        os.makedirs(folder)
        log_path = f'{folder}/{args.algo}({args.model})_{args.data}.log'
        logging.basicConfig(
            level=logging.DEBUG, filename=log_path, filemode='w', format=FORMAT)

        print(f'Algo: {args.algo}, Model: {args.model}, Data: {args.data}, Matrix: {args.matrix}')
        logging.info(
            f'Algo: {args.algo}, Model: {args.model}, Data: {args.data}, Matrix: {args.matrix}')

        print(f'data_path = {data_path}')
        logging.info(f'data_path = {data_path}')

        print(f'data:{len(data)}, y={target} , X_num = {len(X.columns)}')
        logging.info(
            f'data:{len(data)}, y={target} , X_num = {len(X.columns)}')

        print(f'{args.model} config: {model_cfg_json}')
        logging.info(f'{args.model} config: {model_cfg_json}')

        if(args.algo == "PSO"):
            with open('./cfg/algo/PSO_config.json') as f:
                algo_cfg = json.load(f)
                print(f'PSO parameters: {algo_cfg}')
                logging.info(f'PSO config: {algo_cfg}')
            particle_num = algo_cfg['particle_num']
            if (args.model == "MLP"):
                particle_dim = int(len(model_cfg)/2)+2
            else:
                particle_dim = int(len(model_cfg)/2)
            iter_num = algo_cfg['iter_num']
            c1 = algo_cfg['c1']
            c2 = algo_cfg['c2']
            w = algo_cfg['w']

            starttime = datetime.datetime.now()
            pso = PSO(particle_num, particle_dim, iter_num, c1,
                      c2, w, model_cfg, X, y, file, args.model, folder, args.matrix)
            results = pso.main()
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

            virus_num = algo_cfg['virus_num']  # 起始病毒數量
            virus_dim = int((len(model_cfg)-1)/2)  # 超參數調整數量
            s_proportion = algo_cfg['s_proportion']  # 強壯病毒比例
            strong_growth_rate = algo_cfg['strong_growth_rate']  # 強壯病毒複製率
            common_growth_rate = algo_cfg['common_growth_rate']  # 一般病毒複製率
            total_virus_limit = algo_cfg['total_virus_limit']
            intensity = algo_cfg['intensity']
            starttime = datetime.datetime.now()
            voa = VOA(virus_num, virus_dim, strong_growth_rate,
                      common_growth_rate, s_proportion, total_virus_limit, intensity, model_cfg, X, y, args.model, args.matrix)
            best_fitness, best_parameter, lb_fitness, lb_parameter, parms, train, test = voa.main()
            endtime = datetime.datetime.now()
            logging.info(
                (f"best_parameter:{best_parameter},best_fitness:{best_fitness}"))
            logging.info((f"time:{endtime - starttime} "))

            #儲存資料
            parms = pd.DataFrame(parms)
            results = pd.concat(
                [pd.DataFrame(train, columns=["train"]), pd.DataFrame(test, columns=["test"]), parms], axis=1)
            #results.columns = data_columns
            results.to_csv(
                f'{folder}/{args.algo}({args.model})_{file}.csv')
            plt.figure()
            plt.title("VOA Algorithm", fontsize=20)
            plt.xlabel("iteration", fontsize=15, labelpad=15)
            plt.ylabel("f(gb)", fontsize=15, labelpad=20)
            plt.plot(lb_fitness, color='b')
            plt.savefig(
                f'{folder}/VOA_{args.model}_{file}')
            print("finished!!!")

        boxpath = f'{folder}/{args.algo}_{args.model}_{file}_box.jpg'
        plot_boxplot(results, boxpath)
