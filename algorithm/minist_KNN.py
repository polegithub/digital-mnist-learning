from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from data_manager import *

# read_data()
X_train_small, y_train_small, X_test = read_data()

print("x train:")
print(X_train_small.shape)

print("\ny train:")
print(y_train_small.shape)

print("\nx test:")
print(X_train_small.shape)

# knn
# begin time
start = time.clock()
# progressing
knn_clf = KNeighborsClassifier(
    n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
score = cross_val_score(knn_clf, X_train_small, y_train_small, cv=3)

parameters = {'n_neighbors': [3, 5, 7, 9],
              'algorithm': ['kd_tree', 'ball_tree']}

gs_clf = GridSearchCV(knn_clf, parameters, n_jobs=1, verbose=True)
gs_clf.fit(X_train_small.astype('float')/256, y_train_small)
print_grid_mean(gs_clf.grid_scores_)

print("mean:", score.mean())
# end time
elapsed = (time.clock() - start)
print("Time used:", int(elapsed), "s")

"""
结果：(样本数为1000)
grid_scores_:
mean score | scores.std() * 2 | params
0.850      | (+/-0.024)       | {'algorithm': 'kd_tree', 'n_neighbors': 3}
0.850      | (+/-0.024)       | {'algorithm': 'ball_tree', 'n_neighbors': 3}
0.846      | (+/-0.008)       | {'algorithm': 'kd_tree', 'n_neighbors': 7}
0.846      | (+/-0.008)       | {'algorithm': 'ball_tree', 'n_neighbors': 7}
0.843      | (+/-0.033)       | {'algorithm': 'kd_tree', 'n_neighbors': 5}
0.843      | (+/-0.033)       | {'algorithm': 'ball_tree', 'n_neighbors': 5}
0.832      | (+/-0.020)       | {'algorithm': 'kd_tree', 'n_neighbors': 9}
0.832      | (+/-0.020)       | {'algorithm': 'ball_tree', 'n_neighbors': 9}

结论：
1. 不同算法 (kd_tree / ball_tree) 对结果无影响，ball_tree 只是优化了维度灾难的问题
2. n_neighbors 目前结果来看，3 的 mean score最佳，但是 7 的均方差最小。
3. 1000的样本数太少，仅供参考

"""