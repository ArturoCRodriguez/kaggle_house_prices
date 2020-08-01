import scipy.stats as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
from custom_functions import clip_columns
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler, PowerTransformer, StandardScaler


dist = np.random.normal(20,5,1000)
dist = np.append(dist, np.random.normal(100,15,100))
data = pd.DataFrame({'col1':dist})
ohe = OneHotEncoder(sparse = False)
# scaler = RobustScaler(quantile_range=(1,99))
scaler = StandardScaler()
data['col2'] = scaler.fit_transform(data[['col1']])
data['col3'] = PowerTransformer().fit_transform(data[['col1']])

data = clip_columns(data,['col1'],.1,.9)

sns.set_style({'axes.grid' : True})
fig, ax = plt.subplots(3,1,figsize=(10, 10))
sns.distplot(data['col1'], ax=ax[0])
sns.distplot(data['col2'],ax=ax[1])
sns.distplot(data['col3'],ax=ax[2])
# plt.figure(3)
# sns.distplot(data3)
plt.show()

# print(data['col1'].quantile(0.9))

