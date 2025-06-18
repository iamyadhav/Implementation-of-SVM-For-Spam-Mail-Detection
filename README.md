# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: YADHAV GP
RegisterNumber:  212223230247
*/
```

```

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
## Dataset:

![Screenshot 2025-06-05 125416](https://github.com/user-attachments/assets/677c47c5-c0cb-4d39-8403-bcc9245e3c63)


## Train_Test:

![Screenshot 2025-06-05 125429](https://github.com/user-attachments/assets/734abdd3-53d7-4cf3-8730-a41939fb6f6a)


## y_pred

![Screenshot 2025-06-05 125438](https://github.com/user-attachments/assets/1613206e-b9d2-473a-973f-34314c6068cd)


## Accuracy

![Screenshot 2025-06-05 125444](https://github.com/user-attachments/assets/1687e393-6172-4f84-85c5-a40e10745ede)


## Confusion Matrix

![Screenshot 2025-06-05 125448](https://github.com/user-attachments/assets/ebe3d929-916f-4da5-ac00-73b9c6db656e)


## Classification Report

![Screenshot 2025-06-05 125458](https://github.com/user-attachments/assets/c7a6fc92-10dd-4896-be2d-13fa8abd8672)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
