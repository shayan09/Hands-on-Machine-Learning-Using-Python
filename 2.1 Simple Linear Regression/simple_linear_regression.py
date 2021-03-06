import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
print("X and Y")
print(X, "\n\n", Y.reshape(-1,1), "\n")

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("X train and test")
print(X_train, "\n\n", X_test, "\n")
print("Y train and test")
print(Y_train.reshape(-1,1), "\n\n", Y_test.reshape(-1,1), "\n")

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(regressor,"\n")

#Predicting the test set results
Y_pred = regressor.predict(X_test)
print("Y prediction","\n")
print(Y_pred.reshape(-1,1),"\n")

#Visualizing the Training Set Results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the Test Set Results
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, Y_pred, color='black')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
