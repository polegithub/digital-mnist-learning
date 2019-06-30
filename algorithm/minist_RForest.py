import pandas as pd
import numpy as np
import time
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from data_manager import *

# read_data()
X_train_small, y_train_small, X_test = read_data()

# begin time
start = time.clock()
# progressing
parameters = {'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 12, 100]}

rf_clf = RandomForestClassifier(n_estimators=400, n_jobs=4, verbose=1)

gs_clf = GridSearchCV(rf_clf, parameters, n_jobs=1, verbose=True)
gs_clf.fit(X_train_small.astype('int'), y_train_small)
print_grid_mean(gs_clf.grid_scores_)

"""
结果：(样本数为1000)

grid_scores_:
mean score | scores.std() * 2 | params
0.877      | (+/-0.020)       | {'criterion': 'gini', 'max_features': 12}
0.876      | (+/-0.023)       | {'criterion': 'entropy', 'max_features': 12}
0.875      | (+/-0.025)       | {'criterion': 'gini', 'max_features': 'auto'}
0.871      | (+/-0.045)       | {'criterion': 'gini', 'max_features': 100}
0.869      | (+/-0.034)       | {'criterion': 'entropy', 'max_features': 100}
0.866      | (+/-0.025)       | {'criterion': 'entropy', 'max_features': 'auto'}

结论：
1. max_features 目前最佳为 12
2. gini 略优于 entropy, 但并不明显, 其实各项参数的结果都比较接近。 
3. 1000的样本数太少，仅供参考

"""

print()  # end time
elapsed = (time.clock() - start)
print("Time used:", elapsed)
