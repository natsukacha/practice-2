# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:06:45 2018
https://pythondatascience.plavox.info/scikit-learn/scikit-learn%E3%81%A7%E6%B1%BA%E5%AE%9A%E6%9C%A8%E5%88%86%E6%9E%90
irisをtree
@author: SSG_Share_1
"""

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)

predicted = clf.predict(iris.data)


print(predicted)
print(sum(predicted == iris.target) / len(iris.target))

print(iris.data)
print(iris.target)