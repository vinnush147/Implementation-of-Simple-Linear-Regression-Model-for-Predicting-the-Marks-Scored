# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1.Import the standard Libraries. 
### 2.Set variables for assigning dataset values. 
### 3.Import linear regression from sklearn. 
### 4.Assign the points for representing in the graph. 
### 5.Predict the regression for marks by using the representation of the graph. 
### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vinnush Kumar LS
RegisterNumber: 212223230244
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:

## Dataset:
![WhatsApp Image 2024-08-23 at 07 46 45_6cfced44](https://github.com/user-attachments/assets/5cb56a67-d21a-4d07-a1a1-8968f22a5432)

## Head values:
![WhatsApp Image 2024-08-23 at 07 46 56_a89f1788](https://github.com/user-attachments/assets/27fc127f-31cc-4fd6-9ee2-6054555b5cff)

## Tail values:
![WhatsApp Image 2024-08-23 at 07 47 09_4a2dbeed](https://github.com/user-attachments/assets/f0a8d7b4-9049-4a6b-920d-ad38f8dcb797)


## X and Y values:
![WhatsApp Image 2024-08-23 at 07 47 14_52aeb27b](https://github.com/user-attachments/assets/c4f8f35d-d2cf-4487-881d-c96eba32cbb4)

## Prediction values of X and Y:
![WhatsApp Image 2024-08-23 at 07 47 21_69fc638b](https://github.com/user-attachments/assets/b8345417-268b-4a24-90f5-83de2784b272)

## MSE,MAE and RMSE:
![WhatsApp Image 2024-08-23 at 07 47 27_b0ea79c3](https://github.com/user-attachments/assets/ebeb8151-0af9-4006-92fe-b5def1219fd4)

## Training Set:
![WhatsApp Image 2024-08-23 at 08 27 51_a07117b1](https://github.com/user-attachments/assets/e7c241e6-ffc5-4f13-aa2a-9393f626ca55)


## Testing Set:
![WhatsApp Image 2024-08-23 at 08 27 55_a36ea381](https://github.com/user-attachments/assets/bfcce3da-1158-4135-93a2-cf6dc515cc78)








 










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
