import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

df=pd.read_csv('heart.csv')

# Data Preprocessing

# 1. Encoding --> no categorical values

# 2. Scaling --> StandardScaler

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(df)
train_data_standardScaled = standardScaler.transform(df)

# 3. Data Cleansing

# drop null data
df.dropna(inplace=True)

# drop target is not 1 or 0
idx_nm_1 = df[(df['target']!=0) & (df['target']!=1)].index
df=df.drop(idx_nm_1)

# Split train & test dataset
# 'target' is target variable
Y=df['target']
# other data is predictor variables
X=df.iloc[:,0:12]
# seperate the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

# Random Forest Model

forest = RandomForestClassifier(criterion="gini",n_estimators=10,max_depth=100)
forest.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
print("Random Forest's K-Fold Scores")
print(score_k10)
print("Score's accuracy: ",score_k10.mean())
print()
## Logistic Regression Model
## Use the sklearn to fit the train data
from sklearn.linear_model import LogisticRegression
logisticRegr=LogisticRegression(C=1.0,solver="lbfgs",max_iter=200)
logisticRegr.fit(x_train,y_train)



# K-Fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(logisticRegr,x_test,y_test,cv=kfold,scoring="accuracy")
print("LogisticRegression's K-Fold Scores")
print(score_k10)
print("Score's accuracy: ",score_k10.mean())
print()
# SVM Model
# create an SVM classifier model
svm = SVC(C=1.0,kernel='sigmoid',gamma=1.0)
# fit the model to train dataset
svm.fit(x_train,y_train)
# make predictions using the trained model
y_pred=svm.predict(x_test)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(svm,x_test,y_test,cv=kfold,scoring="accuracy")
print("SVM's K-Fold Scores")
print(score_k10)
print("Score's accuracy: ",score_k10.mean())
print()
# evaluate the model
print("Confusion_matrix>>>")
print(confusion_matrix(y_test,y_pred))
print("Classification Report>>>")
print(classification_report(y_test,y_pred))

# print bar chart

# random forest bar chart

from matplotlib import pyplot as plt
scores = []
forest = RandomForestClassifier(criterion="gini",n_estimators=1,max_depth=1)
forest.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

forest = RandomForestClassifier(criterion="gini",n_estimators=10,max_depth=10)
forest.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

forest = RandomForestClassifier(criterion="gini",n_estimators=100,max_depth=100)
forest.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

forest = RandomForestClassifier(criterion="entropy",n_estimators=1,max_depth=1)
forest.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

forest = RandomForestClassifier(criterion="entropy",n_estimators=10,max_depth=10)
forest.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

forest = RandomForestClassifier(criterion="entropy",n_estimators=100,max_depth=100)
forest.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# draw bar chart
ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['cri=gini,est=1,max=1', 'cri=gini,est=10,max=10', 'cri=gini,est=100,max=100', 'cri=entropy,est=1,max=1', 'cri=entropy,est=10,max=10', 'cri=entropy,est=100,max=100'], rotation=30)
plt.bar(range(len(scores)), scores)
plt.title("random forest score")
plt.show()

# Logistic regression bar chart

from matplotlib import pyplot as plt
scores = []
logisticRegr=LogisticRegression(C=0.1,solver="liblinear",max_iter=50)
logisticRegr.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

logisticRegr=LogisticRegression(C=1.0,solver="liblinear",max_iter=100)
logisticRegr.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

logisticRegr=LogisticRegression(C=10.0,solver="liblinear",max_iter=200)
logisticRegr.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

logisticRegr=LogisticRegression(C=0.1,solver="sag",max_iter=50)
logisticRegr.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

logisticRegr=LogisticRegression(C=1.0,solver="sag",max_iter=100)
logisticRegr.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

logisticRegr=LogisticRegression(C=10.0,solver="sag",max_iter=200)
logisticRegr.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# draw bar chart
ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['c=0.1, sol=liblinear, max=50','c=1.0, sol=liblinear, max=100','c=10.0, sol=liblinear, max=200','c=0.1, sol=sag, max=50','c=1.0, sol=sag, max=100','c=10.0, sol=sag, max=200'], rotation=30)
plt.bar(range(len(scores)), scores)
plt.title("logistic regression score")
plt.show()

# SVM bar chart

from matplotlib import pyplot as plt
scores = []
# create an SVM classifier model
svm = SVC(C=0.1,kernel='linear',gamma=0.1)
# fit the model to train dataset
svm.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# create an SVM classifier model
svm = SVC(C=1.0,kernel='linear',gamma=1.0)
# fit the model to train dataset
svm.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# create an SVM classifier model
svm = SVC(C=10.0,kernel='linear',gamma=10.0)
# fit the model to train dataset
svm.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# create an SVM classifier model
svm = SVC(C=0.1,kernel='sigmoid',gamma=0.1)
# fit the model to train dataset
svm.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# create an SVM classifier model
svm = SVC(C=1.0,kernel='sigmoid',gamma=1.0)
# fit the model to train dataset
svm.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# create an SVM classifier model
svm = SVC(C=10.0,kernel='sigmoid',gamma=10.0)
# fit the model to train dataset
svm.fit(x_train,y_train)

# k-fold score
kfold = KFold(n_splits=10,shuffle=True)
score_k10 = cross_val_score(forest,x_test,y_test,cv=kfold,scoring="accuracy")
scores.append(score_k10.mean())

# draw bar chart
ax = plt.subplot()
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['c=0.1, ker=linear, gam=0.1','c=1.0, ker=linear, gam=1.0','c=10.0, ker=linear, gam=10.0','c=0.1, ker=sigmoid, gam=0.1','c=1.0, ker=sigmoid, gam=1.0','c=10.0, ker=sigmoid, gam=10.0'], rotation=30)
plt.bar(range(len(scores)), scores)
plt.title("SVM score")
plt.show()