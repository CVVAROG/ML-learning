import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def euclid(a, b):
    return np.sqrt(np.sum((a - b)**2))

class KNN:
    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclid(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]



X_train = np.array([ [1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

knn = KNN(k=3)
knn.fit(X_train, y_train)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),np.linspace(y_min, y_max, 200))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z).reshape(xx.shape)

cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
cmap_bold = ["red", "blue"]

plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)


for class_value in np.unique(y_train):
    plt.scatter(
        X_train[y_train == class_value, 0],
        X_train[y_train == class_value, 1],
        c=cmap_bold[class_value], label=f"Class {class_value}", edgecolor="k"
    )

plt.legend()
plt.title("KNN decision boundaries (k=3)")
plt.show()




