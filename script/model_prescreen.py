# import all the functions
import numpy as np
import pandas as pd
## CLASSIFICATION FUNCTIONS
# KN
from sklearn.neighbors import KNeighborsClassifier
# SVC
from sklearn.svm import SVC, LinearSVC
# Ensemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
# Linear models
from sklearn.linear_model import LogisticRegression, SGDClassifier
# NN
from sklearn.neural_network import MLPClassifier
# metrics
from sklearn.metrics import recall_score, precision_score
# model selecti
from sklearn.model_selection import cross_val_predict



'''Script for Selecting the most promising models on scikit learn 
with default parameters. You need to have the full scikit stack installed'''

def classification_task(X, y, strat=None,jobs=1):
    '''
    Load and test different classification tasks
    
    Parameters
    ----------
    X: numpy_array
       The feature train test

    y: numpy_array
       The labels train test

    strat: array of labels
            When using a stratified train_test_split
            it could be useful to stratify also on the train set during
            cross validation. (Default None)

    jobs: int
          The number of jobs for parallel processing. (Default 1)

    Returns
    ----------
    df: pandas dataFrame
        return data frame with precision and recall 
        for each model
    '''
# get the average for the score
    if len(np.unique(y))>2:
        average='weighted'
    else:
        average='binary'
# create classifier dictionary
    classifier_dict = {'knb' : KNeighborsClassifier(),
                    'svc' : SVC(random_state=42),
                    'rfc' : RandomForestClassifier(random_state=42),
                    'logit': LogisticRegression(),
                    'sgd' : SGDClassifier(random_state=42), # remeber to shuffle the data here
                    'mlp' : MLPClassifier(random_state=42),
                    'gbc' : GradientBoostingClassifier(random_state=42),
                    'bg' : BaggingClassifier(random_state=42),
                    'lin_svc' : LinearSVC(random_state=42)}
# classifier_name dict
    ID_dict = {'knb' : 'KNeighbors',
                    'svc' : 'SVC',
                    'rfc' : 'RandomForest',
                    'logit': 'LogisticRegression',
                    'sgd' : 'SGD', # remeber to shuffle the data here
                    'mlp' : 'MLP',
                    'gbc' : 'GradientBoosting',
                    'bg' : 'Bagging',
                    'lin_svc' : 'linearSVC'}
# create output
    class_id = []
    prec = []
    recall = []
# start analysis
    cl_all = len(classifier_dict.keys())
    x=1
    for i in classifier_dict.keys():
        if i == 'sgd': # shuffle indexes for 
            np.random.seed(101)
            shuffle_index = np.random.permutation(np.shape(X)[0]) 
            X,y = X[shuffle_index], y[shuffle_index]
        print('Processing', ID_dict[i], 'model', x, 'of', cl_all)
        y_pred = cross_val_predict(classifier_dict[i], X, y, 
                n_jobs=jobs,cv = 10, groups=strat)
        class_id.append(ID_dict[i])
        prec.append(precision_score(y, y_pred, average=average))
        recall.append(recall_score(y, y_pred,average=average))
        x+=1
    report_df = pd.DataFrame({'Classifier_ID':class_id, 'Precision':prec, 'Recall':recall},
                            columns = ['Classifier_ID', 'Precision', 'Recall'])
    return(report_df)
