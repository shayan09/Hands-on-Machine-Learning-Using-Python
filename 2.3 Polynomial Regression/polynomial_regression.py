#Polynomial Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the data
data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

#Not much information to split the dataset into train and test, to make accurate prediction we need as much info as possible

from sklearn.linear_model import LinearRegression
#Fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Visualizing the results
plt.scatter(X, Y, color="red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color="blue")
plt.title("Negotiate Salary")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()