# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Split the data into training and testing sets.
5.convert the text data into a numerical representation using CountVectorizer.
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7.Finally, evaluate the accuracy of the model.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRASANNA V
RegisterNumber: 212223240123
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy

```

## Output:
Result Output

![image](https://github.com/23012647/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568857/887b0d7f-0083-48d4-89ad-d3e8b413aad3)

data.head()

![image](https://github.com/23012647/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568857/6fd06372-cb47-4e6f-8ab1-8aa7fb447b58)

data.info()

![image](https://github.com/23012647/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568857/671c942a-ced7-4c91-a4e0-81e3c3893822)

data.isnull().sum()

![image](https://github.com/23012647/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568857/4741dfff-dcf7-43e2-af2e-9c32937c2444)

Y_prediction Value

![image](https://github.com/23012647/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568857/5a9748ec-df64-4d8a-9f54-7db87b98d844)

Accuracy Value

![image](https://github.com/23012647/Implementation-of-SVM-For-Spam-Mail-Detection/assets/160568857/cc2ca4e7-26ae-4bb0-b52e-fb3a143118fa)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
