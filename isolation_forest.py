#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:10:31 2020

@author: kevin-tete
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder
import re 

 

df = pd.read_csv("/Users/kevin-tete/Downloads/temperature.csv")

df['temperature'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[0:5]))
df['date'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[6:16]))
df['time'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[17:25]))
df['conectiondeviced'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[30:38]))
df = df.drop('temperature;EnqueuedTime;ConnectionDeviceId', axis = 1)


encoder = OneHotEncoder(sparse=False)

df['conectiondeviced']=encoder.fit_transform(df[['conectiondeviced']])
df['time']=pd.to_datetime(df['time'],format='%H:%M:%S')
df['Hour'] = df['time'].dt.hour 
df['minute'] = df['time'].dt.minute 

counts = df.groupby('time')

y = df['temperature']

x = df[['time']]
y = np.array(y)
print(df)
feature_list = list(x.columns)
features = np.array(x)

"""
train_temperature, test_temperature, train_labels, test_labels = train_test_split(features,y,test_size = 0.3,random_state = 42)

print(train_temperature.shape)
print(test_temperature)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_temperature, train_labels)

predictions = rf.predict(test_temperature)


test_labels = test_labels.astype(np.float)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
"""
range = (0, 10) 
bins = 10 
time =df['time']
plt.hist(y, bins, range, color = 'green', 
        histtype = 'bar', rwidth = 0.8)
plt.xlabel("temperature en degrés celsius")
plt.ylabel("occurences")
plt.savefig('fig2.png', dpi = 400)

plt.show()

model1 = IsolationForest(contamination=0.01)
print(model1.fit(df[['temperature','conectiondeviced','Hour','minute']]))


df['scores']=model1.decision_function(df[['temperature','conectiondeviced','Hour','minute']])

df['anomaly']=model1.predict(df[['temperature','conectiondeviced','Hour','minute']])
print (df['anomaly'])
print(df.head(50))
pyplot.plot(df['scores'], df['temperature'],linestyle = 'none', marker = 'o', c = 'lime',
  markersize = 5)
pyplot.xlabel("scores")
pyplot.ylabel("temperatures")
pyplot.savefig('fig1.png', dpi = 400)


plt.legend()
plt.show();
pyplot.show()
print("-----------------")
print(df['anomaly'].value_counts())


