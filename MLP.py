# author: Agnieszka Rutkowska
from sklearn.neural_network import MLPClassifier 
from sklearn import datasets 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from timeit import default_timer as timer

digits = datasets.load_digits(n_class=10) # zbiór przedstawiający cyfry 
n_samples = digits['target'].shape[0] 
X_train = digits['data'][:n_samples//2] # podział danych na pół 
y_train = digits['target'] [:n_samples//2] 
X_test = digits['data'][n_samples//2 :] 
y_test = digits['target'] [n_samples//2:]

clf = MLPClassifier(hidden_layer_sizes=(10,), 
                    solver='lbfgs', 
                    max_iter=1000) # 1 ukryta warstwa z 10 neuronami 

clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

times = []
accuracy_scores = []

for i in range(1,26,2):
    clf = MLPClassifier(hidden_layer_sizes=(i,), 
                        solver='lbfgs', 
                        max_iter=1000)
    
    start = timer() 
    clf.fit(X_train, y_train) 
    end = timer()
    times.append(end-start)
    y_pred = clf.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

x = range(1,26,2)
plt.plot(x,times, '-b', label="Czas uczenia")
plt.plot(x,accuracy_scores, '-r', label="Dokładnosć klasyfikacji")
plt.legend(loc="center right")

plt.title("Wykres zmiany w dokładności klasyfikacji oraz czasu uczenia \nw zależności od liczby neuronów.")
plt.savefig("Wykres.png")
plt.show()
    
