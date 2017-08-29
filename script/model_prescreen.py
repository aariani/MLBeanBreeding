'''Script for Selecting the most promising model on scikit learn 
with default parameters. You need to have the full scikit stack installed'''

def Classification_task(X, y):
    '''
    Load and test different classification tasks
    
    Parameters
    ----------
    X: numpy_array
       The feature train test

    y: numpy_array
       The labels train test

    Returns
    ----------
    df: pandas dataFrame
        return data frame with precision and recall 
        for each model
    '''

