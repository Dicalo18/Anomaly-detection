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
df['time'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[17:19]))
df['conectiondeviced'] = df['temperature;EnqueuedTime;ConnectionDeviceId'].map(lambda x :re.sub('[,\!? ;]', '', x[30:38]))
df = df.drop('temperature;EnqueuedTime;ConnectionDeviceId', axis = 1)
print(df[['temperature','conectiondeviced']])

encoder = OneHotEncoder(sparse=False)

df['conectiondeviced']=encoder.fit_transform(df[['conectiondeviced']])

# # Organisation du dataset en fonction des heures d'entrée 
counts = df.groupby('time')
#size = counts.size() # ici , on compte le nombre de données recues (temperatures ) par heure 
# # dans ce boxplot , on peut se rendre compte qu'il ya certaines fois ou les données sont recues plus d'une fois. 
# #Nous allons étudier un peu plus dans le details si ces données sont des anomalies.
#size.plot(kind = 'bar') 

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
#plt.plot(y,time , 'ro', label = 'actual')
#plt.xticks(rotation = '60'); 
#plt.legend()
# #Organisation du dataset en fonction des dates 
# counts1 = df.groupby('date')
# size = counts1.size()
# size.plot(kind = 'bar')
#on veut voir une anomalie sur les temperatures ???
#on va donc isoler le dataset sur les temperatures
#on va utiliser l'algorithme de random forest
#on calcule la moyenne et l'ecart type des temperatures afin  de determiner une fonction
#densité de probabilité et de se servir de cette fonction pour calculer 
#la probabilité d'existence d'un echantillon
#dans notre cas on essaie de voir si dans le dataset des temperatures , nous avons des temperatures extreme ou minimum par jour , heure 
#X = df.drop(['date'], axis = 1)
model1 = IsolationForest(contamination=0.01)
print(model1.fit(df[['temperature','conectiondeviced']]))


df['scores']=model1.decision_function(df[['temperature','conectiondeviced']])

df['anomaly']=model1.predict(df[['temperature','conectiondeviced']])
print (df['anomaly'])
print(df.head(50))

pyplot.plot(df['scores'], df['temperature'],linestyle = 'none', marker = 'o', c = 'lime',
  markersize = 5)
pyplot.xlabel("scores")
pyplot.ylabel("temperatures")
pyplot.savefig('fig1.png', dpi = 400)
pyplot.show()
print("-----------------")
print(df['anomaly'].value_counts())

"""
from sklearn.datasets import make_blobs
x,y = make_blobs(n_samples=100)
print(x)
"""
