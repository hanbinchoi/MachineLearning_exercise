import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet

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
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=321)

# Elastic Net regression

# Elastic net parameters
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
              }

elastic_net = ElasticNet()

# search the best data
Elastic_regressor=GridSearchCV(elastic_net,parameters,scoring='neg_mean_squared_error',cv=10)
Elastic_regressor.fit(x_train,y_train)

print('ElasticNet best parameters >>')
print(Elastic_regressor.best_params_)
print('ElasticNet best score >>')
print(Elastic_regressor.best_score_)


# In[ ]:
