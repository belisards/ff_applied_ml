import json
import pandas as pd
from sklearn.model_selection import train_test_split
# import mca
import numpy as np
from sklearn.pipeline import Pipeline
import sklearn
import warnings

from pandas.api.types import CategoricalDtype

def load_files(background='../FFChallenge_v5/background.csv',train='../FFChallenge_v5/train.csv',
               meta='metadata.t',nanvalues='keep'):
    
    if nanvalues == 'remove':
        nanvalues = [-9,-8,-7,-6,-5,-3,-2,-1]
    elif nanvalues == 'keep':
        nanvalues = None

    with open(meta, 'r') as f:
        data_types_str = json.load(f)

    dtypedict = {k: CategoricalDtype(v) if isinstance(v, str) and 'CategoricalDtype' in v else v for k, v in data_types_str.items()}

    # print('Loading data...')
    # print(dtypedict.keys())
    background = pd.read_csv(background,low_memory=False, \
                             na_values=nanvalues,\
                                dtype=dtypedict,\
                                usecols=dtypedict.keys()).set_index('challengeID')
    
    train = pd.read_csv(train, sep=',', header=0, index_col=0).dropna(how='all')
    
    data  = background.loc[train.index].join(train)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=4242)
    
    # data_train, data_cv = train_test_split(data_train, test_size=0.1, random_state=422)

    return data_train, data_test
    # return data_train, data_cv, data_test
    

def has_missing(df):
    return df.isnull().values.any()

def replace_non_binary(x):
    if x not in [0, 1]:
        return False
    else:
        return x

def prepare_data(df, target,mi_threshold=0.01):
    
    Y = df[target].dropna()
    X = df.iloc[:, :-6].loc[Y.index]
    
    X = X.loc[:,~X.columns.duplicated()]

    meta = pd.read_csv('metadata.csv', index_col=0)

    targets = ['gpa','grit','materialHardship','eviction','layoff','jobTraining']

    predictors = {target: list(meta[meta[target] > mi_threshold].index) for target in targets}

    X = X[predictors[target]]

    assert X.shape[0] == Y.shape[0]
    assert has_missing(X) == False or has_missing(Y) == False

    return X, Y

def gen_data(train, cv, target=['gpa']):
    
    X_train, y_train = prepare_data(train, target)

    X_cv, y_cv = prepare_data(cv, target)   

    assert X_train.shape[0] == y_train.shape[0]
    assert X_cv.shape[0] == y_cv.shape[0]
    assert X_train.shape[1] == X_cv.shape[1]
    assert has_missing(X_train) == False
    
    return X_train, y_train, X_cv, y_cv

def gen_noise(X):

    shape = X.shape
    X = np.random.randint(2, size=shape)
    # get 1st 10 columns
    X10 = X[:,0:10]

    return X10

def transform_binary(X,binary_features,missingnan=[-9,-8,-7,-6,-5,-3,-2,-1]):

    X = X[binary_features].copy().apply(pd.to_numeric)

    X = X.replace({2: 0})

    X.replace(missingnan, 0, inplace=True)

    # replace nan with 0
    X.fillna(0, inplace=True)

    return X

def gen_noise(X):

    shape = X.shape
    X = np.random.randint(2, size=shape)
    # get 1st 10 columns
    X10 = X[:,0:10]

    return X10

def get_indexcol(X,features):
    # get index of train[numeric_features].columns
    indexes = [X.columns.get_loc(c) for c in features if c in X]
    return indexes

def cols_per_type(X_train,datatype='categorical', meta=pd.read_csv('metadata.csv')):
    Xcols = X_train.columns
    cols = meta[meta['type']==datatype]['new_name'].values
    return [col for col in Xcols if col in cols]


def print_scores(scores):
    for key in scores.keys():
        message = "%s: %0.2f (+/- %0.2f)" % (key, scores[key].mean(), scores[key].std() * 2)
        print(message)

meta = pd.read_csv('metadata.csv', index_col=0)

def cols_per_type(X_train,datatype='categorical', meta=meta):
    Xcols = X_train.columns
    meta = meta.reset_index()
    cols = meta[meta['type']==datatype]['new_name'].values
    return [col for col in Xcols if col in cols]



def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    # https://johaupt.github.io/blog/columnTransformer_feature_names.html
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


def splitfeatname(string):
    try:
        id = string.split('__')[1]
    
    except:
        id = string.split('_')[0]
    return id

def getlabelvar(var):
    return meta[meta.index == var].varlab.values[0]


def transform_binary(X,missingnan=[-9,-8,-7,-6,-5,-3,-2,-1]):

    X = X.copy().apply(pd.to_numeric)

    X = X.replace({2: 0})

    X.replace(missingnan, 0, inplace=True)

    return X
