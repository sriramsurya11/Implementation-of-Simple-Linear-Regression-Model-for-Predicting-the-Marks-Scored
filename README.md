# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:   SRIRAM E
RegisterNumber:  212223040207
*/
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("House vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,Y_pred,color="blue")
plt.title("House vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:


 HEAD VALUES
 
![Screenshot 2025-02-27 181803](https://github.com/user-attachments/assets/d0ba9d8c-d4e1-455c-95c0-2f319c4f2187)

 TAIL VALUES
 
![Screenshot 2025-02-27 181811](https://github.com/user-attachments/assets/e4b8d3dd-b02a-48c0-8182-c073f5818637)

 HOURS VALUES
 
![Screenshot 2025-02-27 181819](https://github.com/user-attachments/assets/34d29bcd-da26-4b85-abd2-e312a7de2101)

 SCORES VALUES
 
![Screenshot 2025-02-27 181830](https://github.com/user-attachments/assets/45553fa6-2a53-482c-a158-fa898847af9d)

Y_PREDICTION

![Screenshot 2025-02-27 183405](https://github.com/user-attachments/assets/14b9849d-285d-4f45-84a6-61b6c973b9a1)

 Y_TEST
 
![Screenshot 2025-02-27 181837](https://github.com/user-attachments/assets/30cf7d72-197b-4927-9671-1c76b97c3a52)

RESULT OF MSE,MAE,RMSE

![Screenshot 2025-02-27 181843](https://github.com/user-attachments/assets/ec9e75a9-8613-4cee-976a-af5a7b84e7ca)

TRAINING SET

![Screenshot 2025-02-27 181902](https://github.com/user-attachments/assets/c47ad2a6-4e43-4dad-b9b4-243ef9ba976a)

TEST SET

![Screenshot 2025-02-27 181909](https://github.com/user-attachments/assets/b649d685-b9fe-4d3f-9de9-c5c210b3ae18)











## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
