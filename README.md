# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.  


## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BHUVANESHWARI S
RegisterNumber: 212222220008
*/

```

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:
![image](https://github.com/user-attachments/assets/40711888-706c-4340-b8c7-a7a4ad18f82f)
![image](https://github.com/user-attachments/assets/3d64b889-0f55-47df-a702-e0f42396c82a)










## DATA DUPLICATE
0
## PRINT DATA
![image](https://github.com/user-attachments/assets/3ea34f57-3806-4dcf-98bf-dcdc68a3a86c)

## DATA_STATUS
![image](https://github.com/user-attachments/assets/55bacee3-58d1-4282-b6ce-c4df191fca11)

## Y_PREDICTION ARRAY
![image](https://github.com/user-attachments/assets/7d8a3ce4-7332-4778-a52f-85d6bb37dd64)

## CONFUSION ARRAY
![image](https://github.com/user-attachments/assets/21e25e9c-6f1f-4af4-8fe2-e7cc0bfaf81b)

## ACCURACY VALUE
![image](https://github.com/user-attachments/assets/4172dd68-7c6b-49b8-bab0-7b73f4b689fc)

## CLASSFICATION REPORT
![image](https://github.com/user-attachments/assets/aff44d1f-bc16-4fb7-9eb5-3af20f1fedbe)

## PREDICTION

![image](https://github.com/user-attachments/assets/1864c7d0-ae62-40ac-b460-4cc1cd7df6a5)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
