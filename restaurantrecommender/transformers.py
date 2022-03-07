'''
Transformers for the Yelp data.
* Column_Selector(colname)
* One_Hot_Encoder(colname, value_type, sparse = True)
'''

import pandas as pd
from sklearn import base
from sklearn.feature_extraction import DictVectorizer # for One_Hot_Encoder

'''
----------------
Functions used in One_Hot_Encoder
----------------- '''

def Value_To_Dict(val):
    return {val:1}

def List_To_Dict(the_list):
    return {category:1 for category in the_list}
    
def Flatten_Dict(d, prekey = ''):
    flat_dict = {}
    for key in d:
        if isinstance(d[key], bool) and d[key]:
            flat_dict.update({prekey+'_'+key:1})
        elif isinstance(d[key], str):
            flat_dict.update({prekey+'_'+key+'_'+d[key]:1})
        elif isinstance(d[key], dict):
            flat_dict.update(Flatten_Dict(d[key], prekey=prekey+'_'+key))
    return flat_dict

''' -------------------
Converts a feature column values into a One-Hot Encoding matrix. If
feature values are lists or (nested) dicts, a column for each list 
entry or dict (sub)key is created.
Inputs: colnames is a string of the column name
        value_type is the type (value, list or dict) of feature values
        sparse indicates whether the matrix is sparse
Dependencies: sklearn.feature_extraction.DictVectorizer
              sklearn.base
------------------- '''
class One_Hot_Encoder(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, colnames, value_type = 'value', sparse = True):
        if value_type == 'value':
            self.apply_function_ = Value_To_Dict
        elif value_type == 'list':
            self.apply_function_ = List_To_Dict
        elif value_type == 'dict':
            self.apply_function_ = Flatten_Dict
        self.colnames_ = colnames
        self.dv_ = DictVectorizer(sparse = sparse)

    def fit(self, X, y = None):
        self.dv_.fit(X[self.colnames_].apply(self.apply_function_))
        return self

    def transform(self, X):
        return self.dv_.transform(X[self.colnames_].apply(self.apply_function_))


''' -------------------
Selects and returns the specified column(s)
Inputs: colnames is a list of column(s) to select
Dependencies: sklearn.base
------------------- '''
class Column_Selector(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, colnames):
        self.colnames_ = colnames

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return pd.DataFrame(X[self.colnames_])