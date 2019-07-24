import xgboost as xgb
import time
from sklearn.grid_search import GridSearchCV
from data_manager import *

# read_data()
X_train_small, y_train_small, X_test = read_data()

# begin time
start = time.clock()
# progressing
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [4],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}
              
parameters = {'objective':['binary:logistic'],
              'learning_rate': [0.05,0.08], #so called `eta` value
              'max_depth': [4,5,6],
              'n_estimators': [5,20], #number of trees, change it to 1000 for better results
              }

xgb_model = xgb.XGBClassifier()
# xgb_model.fit(X_train_small, y_train_small)

gs_clf = GridSearchCV(xgb_model, parameters, n_jobs=1, verbose=True)
gs_clf.fit(X_train_small.astype('float')/256, y_train_small)
print_grid_mean(gs_clf.grid_scores_)