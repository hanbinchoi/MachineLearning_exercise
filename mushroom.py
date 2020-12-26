import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def Purity(data):
    max_ = max(data['cluster'])
    min_ = -1
    number_ = max_ - min_ + 1
    if number_ > 1:
        sum_ = 0
        for i in range(-1, max_ + 1):
            # 클러스터 i 값 데이터 -> temp
            temp = data[data['cluster'] == i]
            # 실제 클래스가 0 인 데이터
            number_0 = len(temp['result'][temp['result'] == 0])
            # 실제 클래스가 1 인 데이터
            number_1 = len(temp['result'][temp['result'] == 1])

            if number_0 > number_1:
                sum_ = sum_ + number_0
            else:
                sum_ = sum_ + number_1

        purity = float(sum_) / float(len(data))

        return purity

    else:  # has an outlier so calculate the about -1 clustering
        a = data['result'].mode()  ## get the mode value
        a = int(a)
        b = len(data['result'][data['result'] == a])
        b = int(b)

        purity = float(b) / float(len(data))

        return purity

data = pd.read_csv("mushrooms.csv")
# Drop dirty data
indexNames = data[(data['stalk-root'] == '?')].index
data.drop(indexNames , inplace=True)
data.fillna(method='ffill',inplace=True)

# Label Encoding
# categorical -> numeric
le = LabelEncoder()
for i in range (data.shape[1]):
    data.iloc[:, i] = le.fit_transform(data.iloc[:, i])

y = data.iloc[:, 0]
X = data.iloc[:, 1:23]

# No Scaling
print("** No Scaling **")

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1','P2']

eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3,5,10,15,20,30,50,100]

# Run DBSCAN
for e in eps:
    for min in min_samples:
        model = DBSCAN(eps=e, min_samples=min).fit(X_principal)
        labels = model.labels_
        X_principal = X_principal.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_principal['cluster'] = labels
        X_principal['result'] = y

        purity = Purity(X_principal)

        print("eps = "+str(e)+", min_samples = "+str(min)+" : "+str(purity))

print()

# Standard Scaling
print("** Standard Scaling **")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1','P2']

eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3,5,10,15,20,30,50,100]


# Run DBSCAN
for e in eps:
    for min in min_samples:
        model = DBSCAN(eps=e, min_samples=min).fit(X_principal)
        labels = model.labels_
        X_principal = X_principal.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_principal['cluster'] = labels
        X_principal['result'] = y

        purity = Purity(X_principal)

        print("eps = "+str(e)+", min_samples = "+str(min)+" : "+str(purity))

print()

# Standard Scaling
print("** MinMax Scaling **")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X) * 10
X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1','P2']

eps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
min_samples = [3,5,10,15,20,30,50,100]


# Run DBSCAN
for e in eps:
    for min in min_samples:
        model = DBSCAN(eps=e, min_samples=min).fit(X_principal)
        labels = model.labels_
        X_principal = X_principal.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_principal['cluster'] = labels
        X_principal['result'] = y

        purity = Purity(X_principal)

        print("eps = "+str(e)+", min_samples = "+str(min)+" : "+str(purity))



