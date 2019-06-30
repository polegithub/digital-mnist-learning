
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from data_manager import *

# read_data()
X_train_small, y_train_small, X_test = read_data()

# LR
# begin time
start = time.clock()
# progressing
"""
lbfgs + l2
"""
# parameters = {'penalty': ['l2'], 'C': [2e-2, 4e-2, 8e-2, 12e-2, 2e-1]}
# lr_clf = LogisticRegression(
#     penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=800,  C=0.2)

"""
结果：(样本数为1000)
grid_scores_:
mean score | scores.std() * 2 | params
0.850      | (+/-0.036)       | {'C': 0.12, 'penalty': 'l2'}
0.848      | (+/-0.034)       | {'C': 0.08, 'penalty': 'l2'}
0.848      | (+/-0.034)       | {'C': 0.2, 'penalty': 'l2'}
0.844      | (+/-0.045)       | {'C': 0.04, 'penalty': 'l2'}
0.839      | (+/-0.055)       | {'C': 0.02, 'penalty': 'l2'}

结论：
1. 整理来看，无太大差异，相对来说，C 的取值 0.12 表现稍微好一点
2. 1000的样本数太少，仅供参考
"""


"""
liblinear + l1
"""
parameters = {'penalty': ['l1'], 'C': [2e0, 2e1, 2e2]}
lr_clf=LogisticRegression(penalty='l1', multi_class='ovr', max_iter=800,  C=4 )
"""
结果：(样本数为1000)
grid_scores_:
mean score | scores.std() * 2 | params
0.826      | (+/-0.035)       | {'C': 2.0, 'penalty': 'l1'}
0.820      | (+/-0.050)       | {'C': 200.0, 'penalty': 'l1'}
0.819      | (+/-0.031)       | {'C': 20.0, 'penalty': 'l1'}

结论： 
1. C越大，均方差越大
2. 不同的 C 对 mean score 差异不大
3. 1000的样本数太少，仅供参考

"""

gs_clf = GridSearchCV(lr_clf, parameters, n_jobs=1, verbose=True)
gs_clf.fit(X_train_small.astype('float')/256, y_train_small)
print_grid_mean(gs_clf.grid_scores_)

# end time
elapsed = (time.clock() - start)
print("Time used:", elapsed)
