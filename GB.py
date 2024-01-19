import numpy as np
import TREE

# идея: с каждым деревом делаем шаг по градиенту ошибки
class GradientBoostingRegressorFromScratch:
    def fit(self, X, y):
        self.trees = []

        for i in range(100):# строим 100 деревьев и ограничиваем максимальную глубену дерева
            tree = TREE.DecisionTreeRegressorFromScratch()
            tree.fit(X, y - self.predict(X))#обучаем на target с вычетом ошибки предыдущих деревьев
            self.trees.append(tree)

    def predict(self, X):
        # необходимо заполнить нулями массив прогнозами всех 100 деревьев для того, чтобы впервый раз метод выдал predict 0
        trees_predictions = np.zeros((len(X), len(self.trees)))

        for i, tree in enumerate(self.trees):#итерируемся по всем деревьям(на первой итерации деревьев еще нет)
            #начиная со второй итерации записываем прогнозы первого дерева в нулевую колонку
            #далее в первой колонке исправляются ошибки из нулевой и тд с каждым деревом
            trees_predictions[:, i] = tree.predict(X) * (1 if i == 0 else 0.1) # (1 if i == 0 else 0.1) - скорость обучения

        return np.sum(trees_predictions, axis=1)
