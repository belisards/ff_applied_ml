import json
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, brier_score_loss, f1_score

from pandas.api.types import CategoricalDtype

meta = pd.read_csv('../metadata/variables.csv', index_col=0)
###########

def cols_per_type(X_train,datatype='categorical', 
                  meta=pd.read_csv('../data/FFMetadata_v10.csv',
                                   low_memory=False)):
    
    Xcols = X_train.columns
    cols = meta[meta['type']==datatype]['new_name'].values
    return [col for col in Xcols if col in cols]


def load_files(meta='../metadata/metadata.json', background='../data/FFChallenge_v5/background.csv',
               train='../data/FFChallenge_v5/train.csv',
               leaderboard='../data/leaderboard.csv',
               holdout='../data/test.csv',
               nanvalues='keep'):
    
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

    leaderboard = pd.read_csv(leaderboard, low_memory=False)
    X_leaderboard = background[background.index.isin(leaderboard.dropna().index)]
    Y_leaderboard = leaderboard.dropna()
    leaderboard = X_leaderboard.join(Y_leaderboard).set_index('challengeID')

    holdout = pd.read_csv(holdout, low_memory=False)
    X_holdout = background[background.index.isin(holdout.dropna().index)]
    Y_holdout = holdout.dropna().copy()
    # Y_holdout['eviction'] = Y_holdout['eviction'].astype('int')
    # Y_holdout['layoff'] = Y_holdout['layoff'].astype('int')
    # Y_holdout['jobTraining'] = Y_holdout['jobTraining'].astype('int')
    holdout = X_holdout.join(Y_holdout).set_index('challengeID')
    
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

    return data_train, data_test, leaderboard, holdout

    # return data_train, data_cv, data_test
    
def has_missing(df):
    return df.isnull().values.any()

def prepare_data(df, target,mi_threshold=0.01):
    
    Y = df[target].dropna()
    X = df.iloc[:, :-6].loc[Y.index]
    
    assert X.shape[0] == Y.shape[0]
    assert has_missing(X) == False or has_missing(Y) == False

    return X, Y

def score_model(model, target, test, leaderboard, holdout, classifier=False):
    X_test, y_test = prepare_data(test, target)

    if classifier:
        # Compute test scores
        y_pred = model.predict(X_test)
        brier = brier_score_loss(y_test, y_pred)
        f1 = f1_score(y_test, y_pred.round())
        
        # Print test scores
        print(f'Test Brier: {brier:.4f}')
        print(f'Test F1: {f1:.4f}')

        # # Compute leaderboard scores
        X_leaderboard, y_leaderboard = prepare_data(leaderboard, target)
        y_pred = model.predict(X_leaderboard)
        brier = brier_score_loss(y_leaderboard, y_pred)
        f1 = f1_score(y_leaderboard, y_pred.round())

        # # Print leaderboard scores
        print(f'Leaderboard Brier: {brier:.4f}')
        print(f'Leaderboard F1: {f1:.4f}')

        # # # Compute holdout scores
        if holdout is not None:
            X_holdout, y_holdout = prepare_data(holdout, target)
            y_pred = model.predict(X_holdout)
            y_holdout = y_holdout.astype(int)
            brier = brier_score_loss(y_holdout, y_pred)
            print(f'Holdout Brier: {brier:.4f}')
            
        
    else:
        score = -model.best_score_ 
        metric_name = model.scorer_._score_func.__name__
        print(f'Metric: {metric_name}')
        
        print(f'Best CV score: {score:.4f}')
        
        # st dev of CV scores
        std_score = model.cv_results_['std_test_score'].mean()
        print(f'Standard deviation of CV scores: {std_score:.4f}')
        # Get mean CV score
        mean_score = -model.cv_results_['mean_test_score'].mean()
        print(f'Mean CV score: {mean_score:.4f}')
        
        # Compute test scores
        mse = mean_squared_error(y_test, model.predict(X_test))
        rsquared = r2_score(y_test, model.predict(X_test))
        
        # Print test scores
        print(f'Test MSE: {mse:.4f}')
        print(f'Test R2: {rsquared:.4f}')

        # # Compute leaderboard scores
        X_leaderboard, y_leaderboard = prepare_data(leaderboard, target)
        mse = mean_squared_error(y_leaderboard, model.predict(X_leaderboard))
        rsquared = r2_score(y_leaderboard, model.predict(X_leaderboard))

        # Print leaderboard scores
        print(f'Leaderboard MSE: {mse:.4f}')
        print(f'Leaderboard R2: {rsquared:.4f}')

        # # # Compute holdout scores
        if holdout is not None:
            X_holdout, y_holdout = prepare_data(holdout, target)
            # X_holdout_transformed = model.best_estimator_.named_steps['preprocessor'].transform(X_holdout)
            mse = mean_squared_error(y_holdout, model.predict(X_holdout))
            rsquared = r2_score(y_holdout, model.predict(X_holdout))

            # # Print holdout scores
            print(f'Holdout MSE: {mse:.4f}')
            print(f'Holdout R2: {rsquared:.4f}')
        
def shap_show(model, alldata, target, n=5):
    X, y = prepare_data(alldata, target)
    model  = model.best_estimator_.fit(X, y)
    Xtransform = model.named_steps['preprocessor'].transform(X)
    exp = shap.TreeExplainer(model.named_steps['regressor'])
    transformer = model.named_steps['preprocessor']
    names = transformer.get_feature_names_out()
    featnames = [splitfeatname(name) for name in names]
    shap_values = exp.shap_values(Xtransform)
    # get top n features
    top_n_idx = np.argsort(np.abs(shap_values).mean(0))[-n:]
    top_n_feat = [featnames[i] for i in top_n_idx]
    # # get questions
    top_n_vars = [meta[meta.index.isin([feat])].varlab.values for feat in top_n_feat]
    # # reverse order
    # top_n_vars = top_n_vars[::-1]
    shap.summary_plot(shap_values, Xtransform, max_display=n, feature_names=featnames)
    return dict(zip(map(tuple, top_n_vars), top_n_feat))

def splitfeatname(string):
    try:
        id = string.split('__')[1]
    
    except:
        id = string.split('_')[0]
    return id

