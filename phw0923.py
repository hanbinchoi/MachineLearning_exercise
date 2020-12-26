import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

df=pd.read_csv('insurance.csv')

# encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df.sex)
df.sex = le.transform(df.sex)

le.fit(df.smoker)
df.smoker = le.transform(df.smoker)

le.fit(df.region)
df.region = le.transform(df.region)
df=df.astype('int')

# split target and predict value
x=df.iloc[:,0:-1]
y=df['charges'];

# split test and train
from sklearn.model_selection import train_test_split, KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=321)

# fit the linear model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV

lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)

# get MSE Value
MSE=cross_val_score(lin_reg,x_test,y_test,scoring='neg_mean_squared_error',cv=10)
mean_MSE=np.mean(MSE)
print('Linear Model score: ')
print(mean_MSE)
print()

# ridge model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

grid_ridge = GridSearchCV(ridge,parameters,cv=10,scoring='neg_mean_squared_error')

# fit the model
grid_ridge.fit(x_train,y_train)

# Print the best value and parameters of Ridge
print("Ridge's best parameter: ")
print(grid_ridge.best_params_)
print("Ridge's best score: ")
print(grid_ridge.best_score_)
print()

# lasso model
from sklearn.linear_model import Lasso
lasso=Lasso()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}

grid_lasso = GridSearchCV(lasso,parameters,cv=10,scoring='neg_mean_squared_error')
grid_lasso.fit(x_train,y_train)


# Print the best score and parameters of Lasso

print("Lasso's best parameter: ")
print(grid_lasso.best_params_)
print("Lasso's best parameter: ")
print(grid_lasso.best_score_)

