import numpy as np
import GB
import solution

np.random.seed(42)
model = GB.GradientBoostingRegressorFromScratch()
model.fit(solution.X_train, solution.y_train)

G_M = int(input("girl = 1 ? boy = 2 : "))
age = int(input("age: "))
weight = int(input("weight: "))
height = int(input("height: "))
activ = float(input("activity: "))
protein = float(input("protein: "))
fats = float(input("fats: "))
carbohydrates = float(input("carbohydrates: "))
calories = float(input("calories: "))

person = np.array([[G_M, age, weight, height, activ, protein, fats, carbohydrates, calories]])
print([round(i, 2) for i in model.predict(person)])

from sklearn.ensemble import GradientBoostingRegressor
sklearn_model = GradientBoostingRegressor()
sklearn_model.fit(solution.X_train, solution.y_train)
print([round(i, 2) for i in sklearn_model.predict(person)])