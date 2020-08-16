#!/usr/bin/env python
# coding: utf-8

# In[37]:


from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets 
import scipy.io.arff as arff
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


# ZADANIE 1
data, meta = arff.loadarff('Leukemia2.arff') 
data = np.array(data.tolist())


# In[3]:


cols = data.shape[1]
X = np.array(data[:,0:cols-1]).astype(np.float)
y = data[:,cols-1]


# In[4]:


selector = VarianceThreshold(threshold=1e5)
X_new = selector.fit_transform(X)


# In[5]:


clf = GaussianNB()
score = cross_val_score(clf, X, y, cv=7)
mean = np.mean(score)


# In[6]:


clf = GaussianNB()
score_new = cross_val_score(clf, X_new, y, cv=7)
mean_new = np.mean(score_new)


# In[7]:


print(f'Średnia przy użyciu niezredukowanej liczby atrybutów: {mean:6.3f}. \nPo redukcji: {mean_new:6.3f}.')


# In[8]:


thresholds = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
scores = []
for value in thresholds:
    selector = VarianceThreshold(threshold=value)
    X_new = selector.fit_transform(X)
    clf = GaussianNB()
    score_new = cross_val_score(clf, X_new, y, cv=7)
    scores.append(np.mean(score_new))

x = range(len(scores))
plt.plot(x, scores, '-b')
horiz_line_data = np.array([mean for i in np.arange(len(x))])
plt.plot(x, horiz_line_data, 'r--') 
plt.xticks(x, thresholds)
plt.title("Wykres zmiany w dokładności klasyfikacji \nw zależnosci od wartości parametru threshold.")
plt.show()


# In[9]:


# ZADANIE 2
X_new = SelectKBest(mutual_info_classif, k=1000).fit_transform(X, y) 

# In[17]:


accuracies = []
for value in np.arange(100, 1100, 100):
    X_new = SelectKBest(mutual_info_classif, k=value).fit_transform(X, y) 
    clf = GaussianNB()
    score_new = cross_val_score(clf, X_new, y, cv=7)
    accuracies.append(np.mean(score_new))

x = range(len(accuracies))
plt.plot(x, accuracies, '-g')
plt.xticks(x, np.arange(100, 1100, 100))
plt.title("Wykres zmiany w dokładności klasyfikacji \nw zależnosci od liczby atrybutów.")
plt.show()


# In[18]:


# ZADANIE 3
digits = datasets.load_digits(n_class=10)
n_samples = digits['target'].shape[0] 
X_train = digits['data'][:n_samples//2]
y_train = digits['target'] [:n_samples//2] 
X_test = digits['data'][n_samples//2 :] 
y_test = digits['target'] [n_samples//2:]


# In[46]:


X_new = SelectKBest(mutual_info_classif, k=64).fit_transform(X_train, y_train)
d = int(math.sqrt(X_new.shape[1]))
plt.imshow(X_new[0].reshape(d, d))


# In[47]:


X_new = SelectKBest(mutual_info_classif, k=36).fit_transform(X_train, y_train)
d = int(math.sqrt(X_new.shape[1]))
plt.imshow(X_new[0].reshape(d, d))


# In[48]:


X_new = SelectKBest(mutual_info_classif, k=25).fit_transform(X_train, y_train)
d = int(math.sqrt(X_new.shape[1]))
plt.imshow(X_new[0].reshape(d, d))


# In[58]:


selector = VarianceThreshold(threshold=-3)
X_new = selector.fit_transform(X_train)
d = int(math.sqrt(X_new.shape[1]))
plt.imshow(X_new[0].reshape(d, d))


# In[59]:


selector = VarianceThreshold(threshold=-8)
X_new = selector.fit_transform(X_train)
d = int(math.sqrt(X_new.shape[1]))
plt.imshow(X_new[0].reshape(d, d))

