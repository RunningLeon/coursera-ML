# %load ../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_seq_items', None)

# %config InlineBackend.figure_formats = {'pdf',}
# % matplotlib
# inline

import seaborn as sns

sns.set_context('notebook')
sns.set_style('white')

data1 = np.loadtxt('ex2data1.txt', delimiter=',')
X = data1[:,0:2]
y = data1[:,2]
m = X.shape[0]
# Add x0 = 1 to first column of X
X_new = np.column_stack((np.ones(X.shape[0]), X))
# Transer from 1-D ndarray to matrix !!!
y_new = np.matrix(y).T
# # print(y[1:5])
# X_pos =  np.array([X[i,:] for i in range(m) if y[i] == 1])
# X_neg =  np.array([X[i,:] for i in range(m) if y[i] == 0])
# assert m == len(X_pos)+len(X_neg)
# # print(m,X_pos.shape,X_neg.shape)
#
# ### plot data
# def plotData():
#     plt.figure(figsize=(10,8))
#     plt.scatter(X_pos[:,0], X_pos[:,1], c='b', marker='+', label='Admitted', linewidths=2)
#     plt.scatter(X_neg[:,0], X_neg[:,1], c='y', marker='o', label='Not Admitted', linewidths=2)
#     plt.xlabel('Exam 1 score')
#     plt.ylabel('Exam 2 score')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# plotData()
# ### scalar implementation of sigmoid function
# def sigmoid(z):
#     return 1/(1 + np.exp(-z))
#
# # builtin vecterized sigmoid function expit in scipy
from scipy.special import expit
# # print(expit([[1,2,3],[4,5,6]]))
#
#
# z = np.linspace(-10,10,100)
# plt.grid(True)
# plt.plot(z, sigmoid(z), label='Sigmoid')
# plt.scatter(0, sigmoid(0), c='r', marker='x')
# plt.title('Sigmoid function')

### define costFunction
# def costFunction(theta, X, y, lambda_=0.):
#     m = len(y)
#     J = 0
#     grad = np.zeros(theta.shape)
#     h = expit(X.dot(theta))
#     J = 1/m*(-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)))
#     grad = 1/m*(X.T.dot(h-y))
#     print(" cs grad shape: {}".format(grad.shape))
#     return (J, grad)

def costFunction(theta, X, y, lambda_=0.):
    m = y.size
    h = expit(X.dot(theta))
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y, lambda_=0.):
    m = y.size
    h = expit(X.dot(theta.reshape(-1,1)))
    grad =(1/m)*X.T.dot(h-y)
    return(grad.flatten())
## check that with theta as zeros, cost is about 0.693
initTheta = np.zeros((X_new.shape[1], 1))
# initTheta = np.matrix(np.zeros((X_new.shape[1], 1)))
# print(initTheta.shape)
cst = costFunction(initTheta, X_new, y_new)
print(cst)

### In octave, use fminunc, here we use scipy.optimize.fmin
from scipy.optimize import fmin, minimize
initial_theta = np.zeros(X.shape[1])
### optimizeTheta
def optimizeTheta(theta, X, y, lambda_=0.):
    res = minimize(costFunction, theta, args=(X, y, lambda_), method=None, jac=gradient, options={'maxiter':400})
#     res = fmin(costFunction, theta, args=(X,y, lambda_), maxiter=400, full_output=True)
    return res

### get opimized theta
result = optimizeTheta(initTheta, X_new, y_new)
print(result)