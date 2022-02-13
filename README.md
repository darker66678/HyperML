# HyperML
Optimizing ML hyperparameter problem using heuristic algorithm
## Support algorithms
### Heuristic
* PSO
* VOA
### Machine Learning
#### Classification
* [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
* [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
#### Regression
* [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
* [MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
* [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
### Scoring
[All the scores](https://scikit-learn.org/stable/modules/model_evaluation.html)
## How to start
### Requirements
Python 3.8

In anaconda, ```conda create --name test python=3.8 ``` can create a new env

```pip install -r requirements.txt``` install all packages
###  Hyperparameter tuning
```python main.py --model [ML_model] --algo [Heuristic] --scoring [score]```
 start to tune the hyperparameters with heuristic

 The results will be saved in ```./results```

