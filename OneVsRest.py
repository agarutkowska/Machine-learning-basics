# author: Agnieszka Rutkowska
from sklearn.linear_model import LogisticRegression
from sklearn import datasets 
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.dummy import DummyClassifier
import numpy as np
import copy

digits = datasets.load_digits(n_class=10) # zbiór przedstawiający cyfry
n_samples = digits['target'].shape[0] 
X_train = digits['data'][:n_samples//2] # podział danych na pół 
y_train = digits['target'] [:n_samples//2] 
X_test = digits['data'][n_samples//2 :] 
y_test = digits['target'] [n_samples//2:]

# Proszę sprawdzić działanie regresji logistycznej na zbiorze digits (10 klas) w dwóch wariantach:
clf = LogisticRegression(max_iter=1500) 
clf.fit(X_train, y_train) 
y_pred1 = clf.predict(X_test)
hidden_ovr_accuracy = accuracy_score(y_test, y_pred1)
print(f'Dokładność klasyfikacji przy użyciu ukrytej klasyfikacji wieloklasowej to {hidden_ovr_accuracy}.')

clf = OneVsRestClassifier(LogisticRegression(max_iter=1500)) 
clf.fit(X_train, y_train) 
y_pred2 = clf.predict(X_test)
ovr_accuracy = accuracy_score(y_test, y_pred2)
print(f'Dokładność klasyfikacji przy użyciu OneVsRestClassifier to {ovr_accuracy}.')

# sprawdzenie dokładności klasyfikatora dummy
clf = DummyClassifier(strategy='most_frequent') 
clf.fit(X_train, y_train) 
y_pred2 = clf.predict(X_test)
dummy_ovr_accuracy = accuracy_score(y_test, y_pred2)
print(f'Dokładność klasyfikacji przy użyciu dummy klasyfikatora to {dummy_ovr_accuracy}.')

# Zadanie napisania własnej klasy OneVsRest
class OneVsRest:
    def __init__(self, model): # przekazanie wcześniej stworzonego modelu
        self.model = model
        self.model_copies = None
        self.classes = None
    
    def fit(self, X, y): # stworzenie tylu kopii modelu i nauczenie każdej
        self.classes = np.unique(y)
        self.model_copies = {}
        
        for class_ in self.classes:
            self.model_copies[class_] = copy.deepcopy(self.model)
            y_tmp = (y == class_)
            self.model_copies[class_].fit(X, y_tmp)
    
    def predict(self, X): # odpowiedź na zasadzie – model dla k-tej klasy zwrócił najwyższe prawdopodobieństwo -> próbka jest klasy k
        prob_values = []
        for class_ in self.classes:
            prob = self.model_copies[class_].predict(X)
            prob_values.append(prob)
        index_max = np.argmax(prob_values) # max(prob_values)
        print(f'Model dla klasy o indeksie {index_max} zwrócił największe prawdopodobieństwo.')
    
    def predict_proba(self, X): # zwraca prawdopodobieństwa (będzie to macierz o wymiarze liczba_próbek_do_predykcji x liczba_klas)
        # tworzenie pustej macierzy do przechowywania prawdobodobieństw
        no_samples = X.shape[0]
        prob_matrix = np.zeros(shape=(no_samples, len(self.classes)))
        for class_ in self.classes:
            for i in range(no_samples):
                prob = self.model_copies[class_].predict(X[i].reshape(1, -1))
                prob_matrix[i][class_] = prob
        return prob_matrix

clf = LogisticRegression(max_iter=1500)
ovr_clf = OneVsRest(clf)
ovr_clf.fit(X_train, y_train)
ovr_clf.predict(X_test[44].reshape(1, -1))
my_matrix = ovr_clf.predict_proba(X_test)
print(my_matrix[44,:])