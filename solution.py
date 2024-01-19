import numpy as np
import pandas as pd
import GB
import TREE
import CHARTS
import warnings
warnings.filterwarnings("ignore")

####################################################sklearn.datasets###################################################
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes

# X = load_breast_cancer()["data"]
# y = load_breast_cancer()["target"]
####################################################house_prices#######################################################
# data = pd.read_csv("data/house_prices.csv")
# print(data)
#
# X = data.drop(["MEDV"], axis=1).values
# y = data["MEDV"].values
####################################################product_NDI########################################################
data = pd.read_csv("data/product_NDI.csv")
print(data)

X = data.drop(["product"], axis=1)
X = X.drop(["target"], axis=1).values
y = data["target"].values
#######################################################################################################################
print(X)
print(X.shape)
print(y)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

np.random.seed(42)
model = GB.GradientBoostingRegressorFromScratch()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

tree = TREE.DecisionTreeRegressorFromScratch()
print(tree.mse(y_test, y_pred))
CHARTS.plot_two_ys(y_pred, y_test)

from sklearn.ensemble import GradientBoostingRegressor
sklearn_model = GradientBoostingRegressor()
sklearn_model.fit(X_train, y_train)
print(tree.mse(y_test, sklearn_model.predict(X_test)))
