import pandas as pd
import numpy as np
def get_resume(data,dependent,independent):
    data_count = data[[independent,dependent]].groupby(independent).count()
    data_mean = data[[independent,dependent]].groupby(independent).mean()
    data_std = data[[independent,dependent]].groupby(independent).std()
    data_var = data[[independent,dependent]].groupby(independent).var()
    data_median = data[[independent,dependent]].groupby(independent).median()
    data_min = data[[independent,dependent]].groupby(independent).min()
    data_max = data[[independent,dependent]].groupby(independent).max()
    data_resume = pd.concat([data_count, data_mean,data_std,data_var,data_median,data_min,data_max],axis=1)
    data_resume.columns = ["count","mean","std","var","median","min","max"]
    return data_resume

def information_gain(data,dependent,independent, norm = False):
    var = data[dependent].var()
    total = data.shape[0]
    var_agg = 0
    data_resume = get_resume(data,dependent,independent)
    for index, row in data_resume.iterrows():
        var_agg += (row['count']/total) * (0 if np.isnan(row['var']) else row['var'])
    result = var - var_agg 
    if norm:
        result = round(result / var,2)
    return result

data = pd.read_csv('train.csv')
feat = "OverallCond"
print("Data completion: {}".format(np.sum(data[feat].count()) / data.shape[0] ))
print("Information gain: {}".format(information_gain(data,"SalePrice",feat, norm=True)))
data[feat].hist()
data.plot(kind="scatter",x=feat, y="SalePrice",alpha=0.1)
print(data[[feat,"SalePrice"]].corr())
get_resume(data,"SalePrice",feat)