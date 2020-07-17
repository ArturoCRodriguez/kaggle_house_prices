#%%
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
data.info()

#%%
# MSSubClass Analysis
data.groupby("MSSubClass").mean()
