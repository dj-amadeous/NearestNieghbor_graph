#imports
from sklearn.datasets import load_iris
import numpy as py
import matplotlib.pyplot as py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
#set iris to iris data set in sk learn
iris = load_iris()
#data and target in iris data set equal to X and y respectively
X = iris.data
y = iris.target
#use train test split method to get an accurate prediction of how model works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

""""for loop changes the value of K, which is the number of nearest neighbors being used to assess the data.
after I go through each iteration, I store the prediction in y_pred.
then after storing the prediction in y_pred, I compare the y_pred to y_test, and get an accuracy score. Then I append that score to the scores array
"""
K_range = range(1, 26)
scores = []
for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

#plot the accuracy on a graph, where number of K neighbors I picked is the x axis, and the accuracy of the prediction is the y axis.
plt.plot(K_range, scores)
plt.xlabel('Value of K splits')
plt.ylabel('Accuracy score for given k splits')
plt.show()
    
