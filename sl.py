from sklearn import datasets
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
plt.show()
# data points colored by species