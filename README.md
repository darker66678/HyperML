# HyperML
Optimizing ML hyperparameter problem using heuristic algorithm
## Updated log
* [Updated log](https://www.dropbox.com/scl/fi/t5tsct6cx9jk9uxznried/HyperML-log.paper?dl=0&rlkey=7qjrdksm1fwmz3ll8ted591za)
## Support algorithms
### Heuristic
* PSO
* VOA
### Machine Learning
#### Classification
* [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
* [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Adaboost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
#### Regression
* [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
* [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
* [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [Adaboost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html?highlight=adaboostre#sklearn.ensemble.AdaBoostRegressor)
### Scoring
[All the scores](https://scikit-learn.org/stable/modules/model_evaluation.html)
## How to start
### Requirements
Python 3.8

In anaconda, ```conda create --name test python=3.8 ``` can create a new env

```pip install -r requirements.txt``` install all packages

### Task
####  Hyperparameter
```python ./hyperml/main.py hyper ```
 start to tune the hyperparameters with heuristic

Please refer following information on parameter config for your experiment
```
hyper_parser.add_argument("-m","--model", help="KNN,MLP,SVM,RF,ADA,XGBoost", default="ADA")
hyper_parser.add_argument("-a","--algo", help="PSO,VOA,RANDOM", default="RANDOM")
hyper_parser.add_argument("-s","--scoring", help="cls: accuracy, f1, recall, precision, specificity; reg: r2, neg_mean_absolute_error, neg_mean_squared_error", default="r2")
hyper_parser.add_argument("-k","--k_fold", help="set k value , need to >1", default=3, type=int)
hyper_parser.add_argument("-c","--confusion_m", help="Do you need to gernerate the confusion_matrix?(False or True)", default=False, type=bool
```

####  Clustering
```python ./hyperml/main.py cluster ```

####  Feature Selection
```python ./hyperml/main.py feat_select ```

 The results will be saved in ```./results```
### Video
[Tutorial](https://drive.google.com/file/d/1sJkAqQfD991WuM9SoE5HFG3WOh9dtvVJ/view?usp=sharing)

