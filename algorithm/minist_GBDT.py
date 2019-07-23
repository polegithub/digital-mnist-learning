from sklearn.ensemble import GradientBoostingClassifier
import time
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from data_manager import *


# batch_x, batch_y, test_x = read_data()
X_train_small, y_train_small, X_test = read_data()

# begin time
start = time.clock()
# progressing
parameters = {'loss': ['deviance'],
              'max_depth':[3,10,2],
              'learning_rate': [1,0.1,0.05]}

rf_clf = GradientBoostingClassifier(n_estimators=100)

gs_clf = GridSearchCV(rf_clf, parameters, n_jobs=1, verbose=True)
gs_clf.fit(X_train_small.astype('float')/256, y_train_small)
print_grid_mean(gs_clf.grid_scores_)

print()  # end time
elapsed = (time.clock() - start)
print("Time used:", elapsed)

"""
'exponential' 会导致：ExponentialLoss requires 2 classes. 参考：https://stackoverflow.com/questions/12197841/why-scikit-gradientboostingclassifier-wont-let-me-use-least-squares-regression
"""

 
