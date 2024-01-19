import numpy as np
import TREE


class GradientBoostingRegressorFromScratch:
    def fit(self, X, y):
        self.trees = []

        for i in range(100):
            tree = TREE.DecisionTreeRegressorFromScratch()
            tree.fit(X, y - self.predict(X))
            self.trees.append(tree)

    def predict(self, X):
        trees_predictions = np.zeros((len(X), len(self.trees)))

        for i, tree in enumerate(self.trees):
            trees_predictions[:, i] = tree.predict(X) * (1 if i == 0 else 0.1)

        return np.sum(trees_predictions, axis=1)
