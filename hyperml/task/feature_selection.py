from sklearn import feature_selection
import logging
import matplotlib.pyplot as plt


def plot_bar(select_type, res, X, y, file):
    bar_X = list(X.columns)
    plt.figure()
    plt.bar(bar_X, res)
    plt.title(f'Data: {file}, Method:{select_type}, Target:{y.name}')
    plt.savefig(f'./results/feature_score/{file}-{select_type}.png')
    


def feat_score(select_type, task_type, y, X, file):
    if(select_type == "chi2"):
        res = feature_selection.chi2(X, y)[0]
    elif(select_type == "pearson"):
        res = feature_selection.r_regression(X, y)
    elif(select_type == "ANOVA"):
        res = feature_selection.f_classif(X, y)[0]
    elif(select_type == "MIC"):
        if(task_type == "classification"):
            res = feature_selection.mutual_info_classif(X, y)
        elif(task_type == "regression"):
            res = feature_selection.mutual_info_regression(X, y)
        else:
            print("unknown task type", task_type)
            return
    print(list(X.columns))
    print(res)
    plot_bar(select_type, res, X, y, file)
    print("finished")
    #TODO record in log