import json
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, brier_score_loss, f1_score, accuracy_score, recall_score, classification_report,roc_auc_score, roc_curve

from pandas.api.types import CategoricalDtype

###########

def cols_per_type(X_train,datatype='categorical', 
                  meta=pd.read_csv('../data/FFMetadata_v10.csv',
                                   low_memory=False)):
    
    Xcols = X_train.columns
    cols = meta[meta['type']==datatype]['new_name'].values
    return [col for col in Xcols if col in cols]



def load_files(meta='../metadata/metadata.json', 
               background='../data/FFChallenge_v5/background.csv',
               train='../data/FFChallenge_v5/train.csv',
               leaderboard='../data/leaderboard.csv',
               holdout='../data/test.csv',
               nanvalues='keep'):
                     
    defaultnan = ['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null']

    if nanvalues == 'remove':
        negative_nanvalues = [-9,-8,-7,-6,-5,-3,-2,-1]
        nanvalues = negative_nanvalues + defaultnan
    
    with open(meta, 'r') as f:
        data_types_str = json.load(f)

    dtypedict = {k: CategoricalDtype(v) if isinstance(v, str) and 'CategoricalDtype' in v else v for k, v in data_types_str.items()}

    # print('Loading data...')
    # print(dtypedict.keys())
    
    # main training data
    background = pd.read_csv(background,low_memory=False, \
                             na_values=nanvalues,\
                             dtype=dtypedict,\
                             usecols=dtypedict.keys()).set_index('challengeID')
    
    train = pd.read_csv(train, sep=',', header=0, index_col=0).dropna(how='all')
    
    data  = background.loc[train.index].join(train)

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

    # leaderboard data
    leaderboard = pd.read_csv(leaderboard, low_memory=False).set_index('challengeID')
    Y_leaderboard = leaderboard.dropna(how='all').copy()
    X_leaderboard = background[background.index.isin(Y_leaderboard.index)]
    leaderboard = X_leaderboard.join(Y_leaderboard)
    
    # holdout
    holdout = pd.read_csv(holdout, low_memory=False).set_index('challengeID')
    Y_holdout = holdout.dropna(how='all').copy()
    X_holdout = background[background.index.isin(Y_holdout.index)]
    holdout = X_holdout.join(Y_holdout)
    
    return data_train, data_test, leaderboard, holdout

def has_missing(df):
    return df.isnull().values.any()

def prepare_data(df, target):
    
    Y = df[target].dropna()
    X = df.iloc[:, :-6].loc[Y.index]
    
    assert X.shape[0] == Y.shape[0]
    assert has_missing(X) == False or has_missing(Y) == False

    return X, Y

def score_model(model, target, test, leaderboard, holdout,classifier=False):
    X_test, y_test = prepare_data(test, target)

    if classifier:
        print('Scores without threshold adjusment')
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
        accuracy = accuracy_score(y_leaderboard, y_pred.round())
        recall = recall_score(y_leaderboard, y_pred.round())

        # r2 = r2_score(y_leaderboard, y_pred)


        # # Print leaderboard scores
        print(f'Leaderboard Brier: {brier:.4f}')
        print(f'Number of positive predictions: {y_pred.sum()}')
        print(f'>> Leaderboard F1: {f1:.4f}')
        print(f'Leaderboard Accuracy: {accuracy:.4f}')
        print(f'Leaderboard Recall: {recall:.4f}')

        # # # Compute holdout scores
        if holdout is not None:
            X_holdout, y_holdout = prepare_data(holdout, target)
            y_pred = model.predict(X_holdout)
            y_holdout = y_holdout.astype(int)
            brier = brier_score_loss(y_holdout, y_pred)
            f1 = f1_score(y_holdout, y_pred.round())
            # r2 = r2_score(y_holdout, y_pred)
            print(f'Holdout Brier: {brier:.4f}')
            print(f'Holdout F1: {f1:.4f}')
            # print(f'Holdout R2: {r2:.4f}')
            
        
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
        print(f'>> Leaderboard MSE: {mse:.4f}')
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

def splitfeatname(string):
    try:
        id = string.split('__')[1]
    except:
        id = string.split('_')[0]
    return id



def score_classifier(model,target,test,leaderboard,holdout=None):
    # load data
    X_test, y_test = prepare_data(test, target)
    X_leaderboard, y_leaderboard = prepare_data(leaderboard, target)
    print('Scores with threshold adjusment')
    
    ## get threshold for optimal FPR
    yhat = model.predict_proba(X_test)
    yhat = yhat[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_test, yhat)
    # calculate AUC
    auc = roc_auc_score(y_test, yhat)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # print('AUC: %.3f' % auc)
    # plot optimal threshold as point
    # find the optimal threshold using youden's j statistic
    ix = np.argmax(tpr - fpr)
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black')
    pyplot.show()
    # print value of optimal threshold
    print('Threshold=%.3f, FPR=%.3f, TPR=%.3f' % (thresholds[ix], fpr[ix], tpr[ix]))
    
    # score test
    yhat = model.predict_proba(X_test)
    yhat = yhat[:, 1]
    yhat = yhat > thresholds[ix]
    # print(classification_report(y_test, yhat))
    brier = brier_score_loss(y_test, yhat)
    print('Test brier: %.3f' % brier)
    f1 = f1_score(y_test, yhat)
    print('Test F1: %.3f' % f1)

    # score leaderboard
    yhat = model.predict_proba(X_leaderboard)
    yhat = yhat[:, 1]
    yhat = yhat > thresholds[ix]
    brier = brier_score_loss(y_leaderboard, yhat)
    print('Leaderboard Brier: %.3f' % brier)
    print(classification_report(y_leaderboard, yhat))
    f1 = f1_score(y_leaderboard, yhat)
    print('Leaderboard F1: %.3f' % f1)

    if holdout is not None:
        X_holdout, y_holdout = prepare_data(holdout, target)
        y_holdout = y_holdout.astype(int)
        yhat = model.predict_proba(X_holdout)
        yhat = yhat[:, 1]
        yhat = yhat > thresholds[ix]
        brier = brier_score_loss(y_holdout, yhat)
        print('Holdout Brier: %.3f' % brier)
        f1 = f1_score(y_holdout, yhat)
        print('Holdout F1: %.3f' % f1)

    

# def get_weights(df,target):
#     weights = df[target].value_counts(normalize=True)
#     w1 = weights[0].round(2)
#     w2 = weights[1].round(2)
#     return [w2, w1]


def feat_2id(feats,meta=pd.read_csv('../metadata/variables.csv',index_col=0)):
    top_n_vars = [meta[meta.index.isin([feat])].varlab.values for feat in feats]
    df = pd.DataFrame(top_n_vars, index=feats, columns=['varlab'])
    return df