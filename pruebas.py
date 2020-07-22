#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
data = pd.DataFrame({'A':[0,1,2,3,4],'B':['cat1','cat2','cat1','cat1','cat3']})
# data['C'] = data.apply(lambda x: 1 if x['A'] != 0 else 0,axis=1)
# print(data.apply(lambda x: 1 if x['A'] != 0 else 0,axis=1))
data

# encoder = OneHotEncoder(sparse=False)
# df = pd.DataFrame(encoder.fit_transform(data), columns=encoder.get_feature_names())
# # encoder.get_feature_names()
# df
pipeline = ColumnTransformer([
    ("nada",'passthrough',['A']),
    ("oh",OneHotEncoder(),['B'])
])
df_transormed = pipeline.fit_transform(data)
# df_final = pd.DataFrame.sparse.from_spmatrix(df_transormed, columns=pipeline.named_transformers_['oh'].get_feature_names())
total_columns = np.append(['A'],pipeline.named_transformers_['oh'].get_feature_names())
df_final = pd.DataFrame(df_transormed, columns=total_columns)
df_final
# # encoder.get_feature_names()
# df

# %%
