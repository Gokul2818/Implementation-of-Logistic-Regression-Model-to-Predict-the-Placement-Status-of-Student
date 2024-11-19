# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data. 
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GOKUL S 
RegisterNumber:  24004336
import pandas as pd
data=pd.read_csv("Placement_Data (1).csv")
print(data.head())
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
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
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print(data1)
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
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
HEAD:

![HEAD](https://github.com/user-attachments/assets/6cca5f9a-80b7-4d51-9a94-c7f4f3d0ca6c)

COPY:

![COPY](https://github.com/user-attachments/assets/642b6f5c-ecd1-46a5-b90c-8bce9bd76cf1)

FIT TRANSFORM:

![FIT TRANSFORM](https://github.com/user-attachments/assets/eb35438f-0a65-4d0d-b3a2-35b8635b005b)

ACCURACY SCORE:

![ACCURACY SCORE](https://github.com/user-attachments/assets/3a197f07-ec53-4653-8e05-d3a81671a90a)

CONFUSION MATRIX:

![CONFUSION MATRIX](https://github.com/user-attachments/assets/39cb718e-ef86-4bbd-9981-80d28593d61b)

CLASSIFICATION REPORT:

![CLASSIFICATION REPORT](https://github.com/user-attachments/assets/f2e6970e-fe23-4d7b-98eb-6f38421332a8)

PREDICTION:

![PREDICTION](https://github.com/user-attachments/assets/9f89e000-e45d-4ede-9b50-91f9b9c0cd8b)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
