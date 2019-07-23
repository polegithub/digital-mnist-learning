
import time
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, NuSVC, SVR
from data_manager import *


# read_data()
X_train_small, y_train_small, X_test = read_data()

# svc
# begin time
start = time.clock()

# progressing

# 1. nuSVC
"""
NuSVC(cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
          max_iter=-1, nu=0.5, probability=False, random_state=None,
          shrinking=True, tol=0.001, verbose=False)
"""
parameters = {'nu': (0.5, 0.02, 0.01), 'gamma': [0.02, 0.01,'auto'],'kernel': ['rbf','sigmoid']}
svc_clf = NuSVC(nu=0.1, kernel='rbf', verbose=0)

'''
结果：(样本数为1000)
grid_scores_:
mean score | scores.std() * 2 | params
0.902      | (+/-0.017)       | {'gamma': 0.02, 'kernel': 'rbf', 'nu': 0.01}
0.901      | (+/-0.016)       | {'gamma': 0.02, 'kernel': 'rbf', 'nu': 0.02}
0.896      | (+/-0.027)       | {'gamma': 0.01, 'kernel': 'rbf', 'nu': 0.02}
0.896      | (+/-0.027)       | {'gamma': 0.01, 'kernel': 'rbf', 'nu': 0.01}
0.888      | (+/-0.040)       | {'gamma': 0.02, 'kernel': 'rbf', 'nu': 0.5}
0.879      | (+/-0.031)       | {'gamma': 0.01, 'kernel': 'rbf', 'nu': 0.5}
0.874      | (+/-0.024)       | {'gamma': 'auto', 'kernel': 'rbf', 'nu': 0.02}
0.872      | (+/-0.019)       | {'gamma': 'auto', 'kernel': 'rbf', 'nu': 0.01}
0.859      | (+/-0.041)       | {'gamma': 'auto', 'kernel': 'rbf', 'nu': 0.5}
0.857      | (+/-0.032)       | {'gamma': 'auto', 'kernel': 'sigmoid', 'nu': 0.02}
0.856      | (+/-0.042)       | {'gamma': 'auto', 'kernel': 'sigmoid', 'nu': 0.5}

结论：
1. rbf 优于 sigmoid
2. gamma: auto的效果并不好, 结果来看最佳为 0.02
3. nu为 0.5 时效果并不好，0.01 和 0.02 时无明显优势，可以考虑加入 0.05 对比测试
4. 1000的样本数太少，仅供参考
'''


# 2. SVC
"""
SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
      coef0=0.0, shrinking=True, probability=False, tol=0.001,
      cache_size=200, class_weight=None, verbose=False, max_iter=-1,
      decision_function_shape='ovr', random_state=None)
"""
# parameters = {'gamma': (0.05, 0.02, 'auto'), 'C': [10, 100, 1.0], 'kernel': ['rbf','sigmoid']}
# svc_clf = SVC(gamma=0.02)

"""
结果：(样本数为1000)
grid_scores_:
mean score | scores.std() * 2 | params
0.901      | (+/-0.016)       | {'C': 10, 'gamma': 0.02, 'kernel': 'rbf'}
0.901      | (+/-0.016)       | {'C': 50, 'gamma': 0.02, 'kernel': 'rbf'}
0.894      | (+/-0.031)       | {'C': 1.0, 'gamma': 0.02, 'kernel': 'rbf'}
0.883      | (+/-0.026)       | {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
0.874      | (+/-0.026)       | {'C': 50, 'gamma': 'auto', 'kernel': 'rbf'}
0.870      | (+/-0.033)       | {'C': 50, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.866      | (+/-0.020)       | {'C': 10, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.864      | (+/-0.019)       | {'C': 10, 'gamma': 0.05, 'kernel': 'rbf'}
0.864      | (+/-0.019)       | {'C': 50, 'gamma': 0.05, 'kernel': 'rbf'}
0.856      | (+/-0.014)       | {'C': 1.0, 'gamma': 0.05, 'kernel': 'rbf'}
0.822      | (+/-0.057)       | {'C': 1.0, 'gamma': 'auto', 'kernel': 'rbf'}
0.754      | (+/-0.053)       | {'C': 1.0, 'gamma': 0.02, 'kernel': 'sigmoid'}

结论：
1. rbf 优于 sigmoid
2. gamma: auto的效果并不好, 结果来看score最佳为 0.02
3. C取10和50甚至100，对mean score无明显影响，但C取1的时候，均方差偏大。
4. 1000的样本数太少，仅供参考
"""

svc_clf.fit(X_train_small.astype('float')/256, y_train_small)

y_test = svc_clf.predict(X_test)
print("\nsvc_clf.score(X_test,y_test):")
print(y_test)
print(svc_clf.score(X_test, y_test))

gs_clf = GridSearchCV(svc_clf, parameters, n_jobs=1, verbose=1)
gs_clf.fit(X_train_small.astype('float')/256, y_train_small)
print_grid_mean(gs_clf.grid_scores_)

# end time
elapsed = (time.clock() - start)
print("Time used:", elapsed)
