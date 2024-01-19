import numpy as np


class DecisionTreeRegressorFromScratch:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.tree_ = {}
        self.max_depth_ = max_depth

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def fit(self, X, y, tree_path='0'):
        if len(tree_path) - 1 == self.max_depth_ or X.shape[0] <= 1:
            self.tree_[tree_path] = np.mean(y)
            return

        minimum_mse = None
        best_split = None

        for feature in range(X.shape[1]):
            for value in sorted(set(X[:, feature])):

                less_than_or_equal_obs = X[:, feature] <= value

                X1, y1 = X[less_than_or_equal_obs], y[less_than_or_equal_obs]
                X2, y2 = X[~less_than_or_equal_obs], y[~less_than_or_equal_obs]

                MSE1 = self.mse(y1, np.mean(y1))
                MSE2 = self.mse(y2, np.mean(y2))
                weight_1 = len(y1) / len(y)
                weight_2 = len(y2) / len(y)
                weighted_mse = MSE1 * weight_1 + MSE2 * weight_2

                if minimum_mse is None or weighted_mse < minimum_mse:
                    minimum_mse = weighted_mse
                    best_split = (feature, value)

        feature, value = best_split
        splitting_condition = X[:, feature] <= value
        X1, y1, X2, y2 = X[splitting_condition], y[splitting_condition], \
                         X[~splitting_condition], y[~splitting_condition]

        self.tree_[tree_path] = best_split

        self.fit(X1, y1, tree_path=tree_path + '0')
        self.fit(X2, y2, tree_path=tree_path + '1')

    def predict(self, X):
        results = []
        for i in range(X.shape[0]):
            tree_path = '0'
            while True:
                value_for_path = self.tree_[tree_path]
                if type(value_for_path) != tuple:
                    result = value_for_path
                    break
                feature, value = value_for_path
                if X[i, feature] <= value:
                    tree_path += '0'
                else:
                    tree_path += '1'
            results.append(result)
        return np.array(results)
