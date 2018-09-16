# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 07:56:04 2018

@author: natsu
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets


digits = datasets.load_digits()

flag = (digits.target==3) + (digits.target==8)

#print(flag)

images = digits.images[flag]
labels = digits.images[flag]
#print(labels)

images = images.reshape(images.shape[0],-1)


#分類
from sklearn import tree

n_samples = len(flag[flag])
train_size = int(n_samples*3/5)
classifier = tree.DecisionTreeClassifier() #分類器作成
classifier.fit(images[:train_size],labels[:train_size])#分類器にデータを与える

print(train_size)
"""
from sklearn import metrics

expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])
print('Accuracy■',metrics.accuracy_score(expected,predicted))
"""