import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
dataset = pd.read_csv("diabetes.csv")
dataset.head()
dataset.info()
dataset.describe()
X = dataset.iloc[:,:1].values
Y = dataset.iloc[:,1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print( regressor.intercept_)
print( regressor.coef_)
Ypred = regressor.predict(X_test)
for (i,j) in zip(Y_test, Ypred):
    if i != j:
        print("Expected Value:",Y_test,"Predicted value:",Ypred)
print("Mislabeled data points in the dataset",(Y_test != Ypred).sum())


