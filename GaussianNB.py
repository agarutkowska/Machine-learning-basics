# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:02:39 2020

@author: arutk
"""

from sklearn.naive_bayes import GaussianNB 
from sklearn import datasets 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score

iris = datasets.load_iris() 
X = iris.data # wartosci cech 
y = iris.target # informacje o klasie
list_accuracy_scores = []
list_f1_scores = []
for i in np.arange(0.1, 1.0, 0.1):
    accuracy_scores = []
    f1_scores = []
    roc_auc_scores = []
    for j in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
        clf = GaussianNB() 
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)
        accuracy_scores.append(adjusted_rand_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    list_accuracy_scores.append(np.mean(accuracy_scores))
    list_f1_scores.append(np.mean(f1_scores))

x = np.arange(0.1, 1.0, 0.1)
plt.plot(x, list_accuracy_scores, '-b', label="adjusted rand score")
plt.plot(x, list_f1_scores, '-r', label="f1 score")
plt.title("Wykres zmiany w dokładności klasyfikacji \nw zależnosci od rozmiaru części uczącej.")
plt.legend(loc="lower left")
plt.show()