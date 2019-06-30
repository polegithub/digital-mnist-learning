import pandas as pd

# read train & test data
def read_data():
    root_path = "digital_minist_data/"
    dataset = pd.read_csv(root_path + "train.csv")
    X_train = dataset.values[0:, 1:]
    y_train = dataset.values[0:, 0]
    # 1000以下的数据量，遍历时间在10s之内，方便测试
    X_train_small = X_train[:1000, :]
    y_train_small = y_train[:1000]
    X_test = pd.read_csv(root_path +"test.csv").values
    return X_train_small, y_train_small, X_test


# handler grid result
def sorted_grid_scores(gridScores):
    def sort_by_mean(val):
        return val.mean_validation_score

    sorted_scores = sorted(gridScores,
                           key=sort_by_mean,
                           reverse=True)
    return sorted_scores



"""
Print grid result

----------
grid_scores_ : list of named tuples

  * ``parameters``, a dict of parameter settings
  * ``mean_validation_score``, the mean score over the cross-validation folds
  * ``cv_validation_scores``, the list of scores for each fold

"""
def print_grid_mean(gridScores, sorted=True):
    print("\ngrid_scores_:")
    print("mean score | scores.std() * 2 | params")   
 
    sorted_scores = gridScores
    if sorted:
        sorted_scores = sorted_grid_scores(gridScores)

    for params, mean_score, scores in sorted_scores:
        print("%0.3f      | (+/-%0.03f)       | %r" % (mean_score, scores.std() * 2, params))
    print()
    