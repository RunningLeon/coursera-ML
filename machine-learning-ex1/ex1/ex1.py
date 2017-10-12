# %load ../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

# %config InlineBackend.figure_formats = {'pdf',}

import seaborn as sns

sns.set_context('notebook')
sns.set_style('white')
data = pd.read_table('ex1data1.txt', delimiter=',', header=0)
print(data.shape)
print(data.head(5))
