from hashlib import new
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris = datasets.load_iris()

# Scope dataset

# print(type(iris))
# print(iris.keys())
# print(iris.data.shape)
# print(iris.feature_names)
# print(iris.target_names)

# 150 samples, 4 features, targets 0 indexed: setosa = [0], versicolor = [1], virginica = [2]

x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
print(df.head())

scatter_view = pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D')
# plt.show()

# Predicting data labels for new data points

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])

new_data = np.array([
    [5.6, 2.8, 3.9, 1.1],
    [5.7, 2.6, 3.8, 1.3],
    [4.7, 3.2, 1.3, 0.2]
    ])

prediction = knn.predict(new_data)
print('Prediction:', prediction)
# Predicts versicolor for first two data points, setosa for the third