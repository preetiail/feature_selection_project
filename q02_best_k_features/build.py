# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn import feature_selection

def percentile_k_features(data,k=20):
    data1=pd.DataFrame(data)
    data2=data1.to_dict()
    X = pd.DataFrame(data.iloc[:,:-1])
    y = pd.DataFrame(data['SalePrice'])
    X1=X.columns.values
    y1=['SalesPrice']
    transformer = feature_selection.SelectPercentile(f_regression,percentile=20).fit_transform(X, y)   
    dataframep=pd.DataFrame(transformer)
    list=['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    return(list)
percentile_k_features(data,k=20)

