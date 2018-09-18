# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:19:32 2018
randomeForest
@author: SSG_Share_1
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets


digits = datasets.load_digits()

flag = (digits.target==3) + (digits.target==8)

#print(flag) see

images = digits.images[flag]
labels = digits.target[flag]
print(images,"■\n\n\n",labels,"■")

images = images.reshape(images.shape[0],-1)



#
from sklearn import ensemble
#from sklearn.tree import DecisionTreeClassifier

n_samples = len(flag[flag])

train_size = int(n_samples*3/5)

classifier = ensemble.RandomForestClassifier(n_estimators=20,max_depth=3,criterion="gini")
digits = datasets.load_digits()#data
#print(digits.data)#see


classifier.fit(images[:train_size],labels[:train_size])#give data #error point

print(train_size)

from sklearn import metrics

expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])
print('Accuracy■',metrics.accuracy_score(expected,predicted),"\n")
print("precision",metrics.precision_score(expected,predicted,pos_label=3),"\n")
print("Recall",metrics.recall_score(expected,predicted,pos_label=3),"\n")
print("F",metrics.f1_score(expected,predicted,pos_label=3),"\n")