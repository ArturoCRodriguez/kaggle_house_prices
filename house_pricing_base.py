import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
data = pd.read_csv('train.csv')
data = data.dropna(axis=1)
X = data.drop('SalePrice',axis=1)
y = data['SalePrice']

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state= 42)
cat_features = []
num_features = []
drop_features = []
for x in X_train.columns:    
    if np.object == X_train[x].dtype:
        cat_features.append(x)
    elif pd.api.types.is_numeric_dtype(X_train[x].dtype):
        num_features.append(x)
    else:
        drop_features.append(x)

full_pipeline = ColumnTransformer([
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat_features),
    ("num",RobustScaler(),num_features)
])
X_train_droped = X_train.drop(drop_features, axis=1)
X_train_prepared = full_pipeline.fit_transform(X_train_droped)

clf = XGBRegressor()
# params = [{"eta":[0.1,0.3,1,3],"gamma":[0,1,3],"max_depth":[3,6,10,60,100],"alpha":[0,1,3,10] }]
# grid = GridSearchCV(clf,params,scoring="neg_mean_squared_log_error", cv=5)
# grid.fit(X_train_prepared,y_train)
# clf = grid.best_estimator_
clf.fit(X_train_prepared,y_train)
X_test_droped = X_test.drop(drop_features, axis=1)
X_test_prepared = full_pipeline.transform(X_test_droped)
y_pred = clf.predict(X_test_prepared)

error = math.sqrt(mean_squared_log_error(y_test,y_pred))
print(error)






    