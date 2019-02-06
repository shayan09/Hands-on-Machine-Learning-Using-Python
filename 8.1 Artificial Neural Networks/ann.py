# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset)
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])
le_2 = LabelEncoder()
X[:, 2] = le_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #to avoid the dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling - compulsory for ANN's
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

import keras
import tensorflow as tf
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} ) #personal device setup for training
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
#Add the input and first hidden layer
model.add(Dense(units=6, activation= 'relu', kernel_initializer='uniform', input_dim = 11))

#Add the second hidden layer
model.add(Dense(units=6, activation= 'relu', kernel_initializer='uniform'))

#Adding the output layer
model.add(Dense(units=1, activation= 'sigmoid', kernel_initializer='uniform')) #use softmax if there are more than 2 categories

#Compile the ANN
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
model.fit(X_train, Y_train, batch_size=64, epochs=50)

score, acc = model.evaluate(X_test, Y_test,
                            batch_size=32, verbose=1)
print('Test accuracy:', acc*100,'%')