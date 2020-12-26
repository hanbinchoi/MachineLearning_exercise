import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

df=pd.read_csv('mnist.csv')

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
idx_nm_1 = df[(df['label']<0) & (df['label']>9)].index
df=df.drop(idx_nm_1)

# seperate the data into d1,d2
d1,d2 = train_test_split(df,test_size=0.1)


# 'label' is target variable
Y=d1['label']
# other data is predictor variables
X=d1.iloc[:,1:785]

# seperate the d1 into train and test
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

# compute RandomForest score
def getRFScore(cri,est,max):
    forest = RandomForestClassifier(criterion=cri, n_estimators=est, max_depth=max)
    forest.fit(x_train, y_train)

    # k-fold score
    kfold = KFold(n_splits=10, shuffle=True)
    score_k10 = cross_val_score(forest, x_test, y_test, cv=kfold, scoring="accuracy")

    return score_k10.mean()


# find max score of Random Forset
print("criterion=gini, n_estimators=10, max_depth=10")
print(getRFScore("gini",10,10))
print()
print("criterion=gini, n_estimators=10, max_depth=100")
print(getRFScore("gini",10,100))
print()
print("criterion=gini, n_estimators=100, max_depth=10")
print(getRFScore("gini",100,10))
print()
print("criterion=entropy, n_estimators=10, max_depth=10")
print(getRFScore("entropy",10,10))
print()
print("criterion=entropy, n_estimators=10, max_depth=100")
print(getRFScore("entropy",10,100))
print()
print("criterion=entropy, n_estimators=100, max_depth=10")
print(getRFScore("entropy",100,10))
print()

print("max score -> criterion=gini, n_estimators=100, max_depth=10")
print()

# compute LogisticRegression score
def getLRScore(C,sol,max):
    logisticRegr = LogisticRegression(C=C, solver=sol, max_iter=max)
    logisticRegr.fit(x_train, y_train)

    # K-Fold score
    kfold = KFold(n_splits=10, shuffle=True)
    score_k10 = cross_val_score(logisticRegr, x_test, y_test, cv=kfold, scoring="accuracy")

    return score_k10.mean()

# find max score of LogisticRegression
print("C=1.0, Solver=liblinear, max_iter=200")
print(getLRScore(1.0,"liblinear",200))
print()
print("C=0.1, Solver=liblinear, max_iter=200")
print(getLRScore(0.1,"liblinear",200))
print()
print("C=0.01, Solver=sag, max_iter=200")
print(getLRScore(1.0,"sag",200))
print()
print("C=0.1, Solver=sag, max_iter=200")
print(getLRScore(0.1,"liblinear",200))
print()
print("C=1.0, Solver=liblinear, max_iter=200")
print(getLRScore(0.1,"liblinear",100))
print()

print("max score -> C=1.0, Solver=liblinear, max_iter=200")
print()


# Compute SVM score
def getSVMScore(c,ker,gam):
    # SVM Model
    # create an SVM classifier model
    svm = SVC(C=c, kernel=ker, gamma=gam)
    # fit the model to train dataset
    svm.fit(x_train, y_train)
    # make predictions using the trained model

    # k-fold score
    kfold = KFold(n_splits=10, shuffle=True)
    score_k10 = cross_val_score(svm, x_test, y_test, cv=kfold, scoring="accuracy")

    return score_k10.mean()

# find max score of SVM
print("C=0.1, Solver=linear, gamma=10.0")
print(getSVMScore(0.1,"linear",10.0))
print()
print("C=1.0, Solver=linear, gamma=1.0")
print(getSVMScore(1.0,"linear",1.0))
print()
print("C=1.0, Solver=linear, gamma=10.0")
print(getSVMScore(1.0,"linear",10.0))
print()
print("C=0.1, Solver=linear, gamma=1.0")
print(getSVMScore(0.1,"linear",1.0))
print()
print("C=1.0, Solver=sigmoid, gamma=1.0")
print(getSVMScore(1.0,"sigmoid",1.0))
print()
print("C=0.1, Solver=sigmoid, gamma=1.0")
print(getSVMScore(0.1,"sigmoid",1.0))
print()

print("max score -> C=0.1, Solver=linear, gamma=10.0")
print()



# get confusion matrix
# 1. Random Forest.
forest = RandomForestClassifier(criterion="gini", n_estimators=100, max_depth=10)
forest.fit(x_train, y_train)

y_pred = forest.predict(x_test)
plt.figure(figsize=(7,5))
plt.title('RandomForest Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.show()

# 2. Logistic Regression
logisticRegr = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
logisticRegr.fit(x_train, y_train)

y_pred = logisticRegr.predict(x_test)
plt.figure(figsize=(7,5))
plt.title('LogisticRegression Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.show()

# 3. SVM

svm = SVC(C=0.1, kernel="linear", gamma=10.0)
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
plt.figure(figsize=(7,5))
plt.title('SVM Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.show()


# Start Ensemble classifier model using D2 -> SVM(C=1.0, solver=linear, gamma=10.0) (high score)

# 'label' is target variable
Y=d2['label']
# other data is predictor variables
X=d2.iloc[:,1:785]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import BaggingClassifier
estimator = SVC(C=1.0, kernel="linear", gamma=10.0)
n_estimators = 10
n_jobs = 1
model = BaggingClassifier(base_estimator=estimator,
                          n_estimators=n_estimators,
                          max_samples=1./n_estimators,
                          n_jobs=n_jobs)
model.fit(x_train,y_train)

score= model.score(x_test,y_test)
print("Ensemble score (Using Bagging)")
print(score)

y_pred = model.predict(x_test)
plt.figure(figsize=(7,5))
plt.title('Ensemble Confusion Matrix')
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.show()