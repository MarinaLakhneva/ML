import numpy as np

#######################################################1###############################################################
# class DecisionTreeRegressorFromScratch:
#   #метод обучения без какого либо обучения
#   def fit(self, X, y):
#     pass
#
#   #метрика качества: Mean Squared Error
#   def mse(self, y_true, y_pred):
#     return np.mean(np.power(y_true - y_pred, 2)) #возводим в квадрат, потому что не хотим больше/меньше прогнозировать
#
#   def predict(self, X):
#     return np.zeros(X.shape[0]) #заполняем массив 0 (X.shape[0] - количество столбцов)
#######################################################2###############################################################
# class DecisionTreeRegressorFromScratch:
#   def fit(self, X, y):
#     self.mean_ = np.mean(y)
#
#   # метрика качества: Mean Squared Error
#   def mse(self, y_true, y_pred):
#     return np.mean(np.power(y_true - y_pred, 2))  # возводим в квадрат, потому что не хотим больше/меньше прогнозировать
#
#   def predict(self, X):
#     return np.repeat(self.mean_, X.shape[0])
#######################################################3###############################################################
# class DecisionTreeRegressorFromScratch:
#     def __init__(self):
#         self.tree_ = {}
#
#     def mse(self, y_true, y_pred):
#         return np.mean(np.power(y_true - y_pred, 2))
#
#     def fit(self, X, y, tree_path='0'):
#         minimum_mse = None
#         self.best_feature_ = None
#         self.best_value_ = None
#
#         #находим лучшее разделение на две группы
#         for feature in range(X.shape[1]): #итерируемся по всем фичам в датасете
#             for value in X[:, feature]: #итерируемся по всем значениям внутри фичи
#                 #создаем булев массив less_than_or_equal_obs, который содержит True для всех элементов массива X[:, feature],
#                 # которые меньше или равны заданному значению value, и False для всех остальных элементов
#                 less_than_or_equal_obs = X[:, feature] <= value
#                 X1, y1 = X[less_than_or_equal_obs], y[less_than_or_equal_obs]
#                 X2, y2 = X[~less_than_or_equal_obs], y[~less_than_or_equal_obs]
#
#                 MSE1 = self.mse(y1, np.mean(y1))
#                 MSE2 = self.mse(y2, np.mean(y2))
#                 weight_1 = len(y1) / len(y)
#                 weight_2 = len(y2) / len(y)
#                 weighted_mse = MSE1 * weight_1 + MSE2 * weight_2
#
#                 if minimum_mse is None or weighted_mse < minimum_mse:
#                     minimum_mse = weighted_mse
#                     self.best_feature_ = feature
#                     self.best_value_ = value
#
#         final_cond = X[:, self.best_feature_] >= self.best_value_
#         self.mean_1 = np.mean(y[final_cond])
#         self.mean_2 = np.mean(y[~final_cond])
#
#     def predict(self, X):
#         n_rows = len(X)
#         results = []
#         for i in range(n_rows):
#             if X[i, self.best_feature_] >= self.best_value_:
#                 result = self.mean_1
#             else:
#                 result = self.mean_2
#             results.append(result)
#         return np.array(results)
#######################################################4###############################################################
class DecisionTreeRegressorFromScratch:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.tree_ = {}#словарь условий, которые мы применяем
        self.max_depth_ = max_depth

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def fit(self, X, y, tree_path='0'):
        # условие выхода: либо дошли до конца, либо когда мы находимся на последнем наблюдении
        if len(tree_path) - 1 == self.max_depth_ or X.shape[0] <= 1:
            self.tree_[tree_path] = np.mean(y)
            return

        minimum_mse = None
        best_split = None
        #1 находим лучшее разделение
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

        #2: нашли разделение и сразу его применили
        feature, value = best_split
        splitting_condition = X[:, feature] <= value
        X1, y1, X2, y2 = X[splitting_condition], y[splitting_condition], \
                         X[~splitting_condition], y[~splitting_condition]

        self.tree_[tree_path] = best_split

        #3 рекурсивно вызываем метод обучения
        self.fit(X1, y1, tree_path=tree_path + '0')
        self.fit(X2, y2, tree_path=tree_path + '1')

    def predict(self, X):
        results = []
        for i in range(X.shape[0]):
            tree_path = '0'
            while True:
                value_for_path = self.tree_[tree_path]
                if type(value_for_path) != tuple:#выходим когда достигнем значение float, а не кортеж
                    result = value_for_path
                    break
                feature, value = value_for_path
                if X[i, feature] <= value:
                    tree_path += '0'
                else:
                    tree_path += '1'
            results.append(result)
        return np.array(results)
