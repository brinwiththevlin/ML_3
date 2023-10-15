import pandas as pd
import plotly.express as px
def accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """returns the accuracy of the model

    Args:
        y_true (pd.Series): treu lables
        y_pred (pd.Series): predicted labels

    Returns:
        float: accuracy [0-100]
    """
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN =  ((y_true == 0) & (y_pred == 0)).sum()
    return TP+TN/len(y_true)*100
    
def accuracy_plot(): 
    pass


def F1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """calculate F1 score given true labels and predicted labels

    Args:
        y_true (pd.Series): true labels
        y_pred (pd.Series): predicted labels

    Returns:
        float: f1 score [0-1] 1 being a perfect score
    """
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP =  ((y_true == 0) & (y_pred == 1)).sum()
    FN =  ((y_true == 1) & (y_pred == 0)).sum()
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    
    f1 = 2*(precision*recall)/(precision+recall)
    return f1



#TODO: decide what the input should look like
def F1_plot():
    pass

# def 

def ROC_plot():
    pass