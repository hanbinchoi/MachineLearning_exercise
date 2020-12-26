import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

fever = ['L', 'M', 'H', 'M','H',
         'L', 'H', 'M', 'M','H',
         'L', 'M', 'H', 'M','H',
         'H', 'M', 'H', 'L','L',
         'M', 'L', 'L', 'M','H',
         'H', 'H', 'L', 'L','H']

sinus = ['Y', 'N', 'N', 'N','Y',
         'N', 'N', 'Y', 'N','Y',
         'Y', 'Y', 'Y', 'Y','N',
         'N', 'Y', 'N', 'Y','N',
         'Y', 'N', 'Y', 'Y','Y',
         'N', 'Y', 'N', 'Y','N']

ache = ['Y', 'N', 'Y', 'N','N',
        'N', 'Y', 'Y', 'N','N',
        'Y', 'N', 'Y', 'Y','N',
        'N', 'N', 'N', 'N','N',
        'N', 'Y', 'N', 'Y','Y',
        'Y', 'Y', 'N', 'Y','Y']

swell = ['N', 'Y', 'Y', 'N','N',
         'Y', 'Y', 'Y', 'N','Y',
         'N', 'Y', 'N', 'N','N',
         'N', 'N', 'Y', 'Y','N',
         'Y', 'N', 'Y', 'Y','Y',
         'Y', 'N', 'Y', 'Y','N']

headache = ['N', 'Y', 'Y', 'Y','N',
            'Y', 'N', 'N', 'Y','N',
            'Y', 'N', 'N', 'Y','N',
            'N', 'N', 'Y', 'Y','Y',
            'N', 'Y', 'Y', 'N','N',
            'Y', 'N', 'N', 'N','Y']

flu = ['Y', 'N', 'Y', 'Y', 'N',
       'Y', 'N', 'N', 'Y', 'Y',
       'Y', 'Y', 'Y', 'Y', 'N',
       'N', 'Y', 'Y', 'Y', 'Y',
       'N', 'Y', 'N', 'N', 'Y',
       'N', 'N', 'Y', 'N', 'N']

# Label Encoding
fever_le = preprocessing.LabelEncoder()
fever_encoded = fever_le.fit_transform(fever)

sinus_le = preprocessing.LabelEncoder()
sinus_encoded = sinus_le.fit_transform(sinus)

ache_le = preprocessing.LabelEncoder()
ache_encoded = ache_le.fit_transform(ache)

swell_le = preprocessing.LabelEncoder()
swell_encoded = swell_le.fit_transform(swell)

headache_le = preprocessing.LabelEncoder()
headache_encoded = headache_le.fit_transform(headache)

label_le = preprocessing.LabelEncoder()
label = label_le.fit_transform(flu)

# set features
features = []

for x,y,z,w,v in zip(fever_encoded,sinus_encoded,ache_encoded,swell_encoded,headache_encoded):
    features.append((x,y,z,w,v))


# Train the model using the training sets
model = GaussianNB()
model.fit(features,label)

# predict
f = [0,1,2]
s = [0,1]
a = [0,1]
sw = [0,1]
h = [0,1]
print("(fever, sinus, ache, swell, headache) -> flu")
for x in f:
    for y in s:
        for z in a:
            for w in sw:
                for v in h:
                    predicted = model.predict([[x,y,z,w,v]])
                    print("({0}, {1}, {2}, {3}, {4}) -> {5}".format(fever_le.inverse_transform([x])[0],
                          sinus_le.inverse_transform([y])[0],
                          ache_le.inverse_transform([z])[0],
                          swell_le.inverse_transform([w])[0],
                          headache_le.inverse_transform([v])[0],
                          label_le.inverse_transform(predicted)[0]))