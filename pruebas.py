#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
data = pd.DataFrame({
    'A':[0,1,2,3,4],
    'B':['cat1','cat2','cat1','cat1','cat3'],
    'C':['c4','c1','c3','c3','c2']
    })
# data['C'] = data.apply(lambda x: 1 if x['A'] != 0 else 0,axis=1)
# print(data.apply(lambda x: 1 if x['A'] != 0 else 0,axis=1))
data

# encoder = OneHotEncoder(sparse=False)
# df = pd.DataFrame(encoder.fit_transform(data), columns=encoder.get_feature_names())
# # encoder.get_feature_names()
# df
pipeline = ColumnTransformer([
    ("nada",'passthrough',['A']),
    ("oh",OneHotEncoder(),['B','C'])
])
df_transormed = pipeline.fit_transform(data)
# df_final = pd.DataFrame.sparse.from_spmatrix(df_transormed, columns=pipeline.named_transformers_['oh'].get_feature_names())
total_columns = np.append(['A'],pipeline.named_transformers_['oh'].get_feature_names())
df_final = pd.DataFrame(df_transormed, columns=total_columns)
df_final
# # encoder.get_feature_names()
# df

# %%
from random import randint
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 

values_dict = {}
values_dict['A'] = []
values_dict['B'] = []
values_dict['C'] = []
for x in range(200):
    values_dict['A'].append(randint(0,99))
    values_dict['B'].append('cat'+str(randint(1,10)))
    values_dict['C'].append('fig'+str(randint(1,20)))
my_df = pd.DataFrame(values_dict)
my_df

my_total_columns = np.append(['A'],pipeline.named_transformers_['oh'].get_feature_names())
my_df_transformed = pd.DataFrame.sparse.from_spmatrix(pipeline.fit_transform(my_df), columns= my_total_columns)
my_df_transformed
    
    