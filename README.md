# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Yadhav G P
RegisterNumber: 212223230247
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
*/
```

## Output:
## Encoding"
![443017186-5c4b1492-8367-4f3f-9f6b-e99238db16e2](https://github.com/user-attachments/assets/c7ad5b3d-27b3-4d41-8d48-9d17ccda28a3)
## Head():
![443017314-97a34e8e-99da-4641-834d-7d23fc76ecd1](https://github.com/user-attachments/assets/89b89b27-e0d4-4904-823a-f4c49058ff8a)
## Info():
![443017423-f76ada16-196c-489b-968f-8c05fa080674](https://github.com/user-attachments/assets/2473802c-6cf5-4d63-8aa2-028aafd6e8e9)

## isnull().sum():
![443017573-740db595-cedd-44bd-9b28-72282aaf4bec](https://github.com/user-attachments/assets/62bfae50-78aa-4c27-a198-7a2490d5827f)
## Prediction of y:
![443018401-853489b2-9a35-4319-847d-d9eb45c1192a](https://github.com/user-attachments/assets/5e2adeef-5cce-4839-ad77-95e7e5158e86)

## Accuray:
![443018580-8eee9fd1-1457-4e40-a1df-44f2fa714a3d](https://github.com/user-attachments/assets/d3a8beba-f999-4ac6-8288-6b4fbdeb1f78)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
