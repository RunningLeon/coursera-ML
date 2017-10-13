# %load ../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

### Read data1
data1 = pd.read_csv('ex1data1.txt', header=None, names=['X', 'y'])
# data1.plot.scatter(x='X',y='y', c='r',marker='x')

# print(data1.shape)
# print(data1.head(5))
# X = data1['X'].values
# y = data1['y'].values
X = np.array(data1['X'])
y = np.array(data1['y'])
# Scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, c='r', marker='x', linewidths=1)
#
# plt.show()
# plt.grid(True)
# plt.xlabel('Profit in $10,000s')
# plt.ylabel('Population of City in 10,000s')

### compute cost function
iterations = 1500
alpha = 0.01
# print(data1.shape)

def computeCost(X, y, theta=None ):
    if theta == None:
        theta = [[0],[0]]
    m = y.size
    return 1/(2*m)*np.sum(np.square(X.dot(theta) - y))
i = 0
for x in X:
    i += 1
    # print("row({}): {}".format(i, x))

X = np.c_(np.ones(X.shape[0]), X)

print(computeCost(X, y))
