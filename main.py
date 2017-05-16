from sklearn.datasets import load_iris
import numpy as py
import matplotlib.pyplot as py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

K_range = range(1, 26)
scores = []
for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
    
plt.plot(K_range, scores)
plt.xlabel('Value of K splits')
plt.ylabel('Accuracy score for given k splits')
plt.show()
    
