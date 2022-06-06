# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Start the program
2. Import the libraries and the Datasets
3. And then split the datasets into the training set and test set.
4. Then import the StandardScaler and Fit the LogisticRegression into the Training set
5. After fitting , Predict the test set results
6. Import the ConfusionMatrix and metrics for making ConfusionMatrix
7. Finally visualize the training set results by importing 'ListedColormap'.
8. Stop the Program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S Saichandran
RegisterNumber:  212220040138
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from re import X
datasets = pd.read_csv('/content/Social_Network_Ads (1).csv')
X=datasets.iloc[:,[2,3]].values
Y=datasets.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_X
StandardScaler()

X_Train=sc_X.fit_transform(X_Train)
X_Test=sc_X.transform(X_Test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)

LogisticRegression(random_state=0)
Y_Pred=classifier.predict(X_Test)
Y_Pred

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_Test,Y_Pred)
cm

from sklearn import metrics
accuracy = metrics.accuracy_score(Y_Test,Y_Pred)
accuracy 

recall_sensitivity = metrics.recall_score(Y_Test,Y_Pred,pos_label=1)
recall_specificity = metrics.recall_score(Y_Test,Y_Pred,pos_label=0)
recall_sensitivity ,recall_specificity

from matplotlib.colors import ListedColormap
X_Set , Y_Set = X_Train , Y_Train
X1,X2 = np.meshgrid(np.arange(start=X_Set[:,0].min()-1,stop=X_Set[:,0].max()+1,step=0.01),np.arange(start=X_Set[:,1].min()-1,stop=X_Set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=076cmap=ListedColormap(('Yellow','Blue')))
plt.xlim(X1.min(),X2.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_Set)):
  plt.scatter(X_Set[Y_Set==j,0],X_Set[Y_Set==j,1],c=ListedColormap(('White','green'))(i),label=j)
  plt.title('LogisticRegression (Training Set)')
  plt.xlabel('Age')
  plt.ylabel('Estimated Salary')
  plt.legend()
  plt.show()
```
## Output:
![logistic regression using gradient descent](/predictiong%20the%20test%20set%20results.PNG)
![logistic regression using gradient descent](/Making%20the%20confusion%20matrix.PNG)

![logistic regression using gradient descent](/accuracy.PNG)


![logistic regression using gradient descent](/recall.PNG)

![logistic regression using gradient descent](/output%201.PNG)

![logistic regression using gradient descent](/output%202.PNG)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

