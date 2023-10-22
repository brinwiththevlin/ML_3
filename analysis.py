import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from typing import List

plotly.io.renderers.default = "browser"


def accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """returns the accuracy of the model

    Args:
        y_true (pd.Series): true labels
        y_pred (pd.Series): predicted labels

    Returns:
        float: accuracy [0-100]
    """

    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    return TP + TN / len(y_true) * 100


def accuracy_plot(accuracies: np.ndarray, model: str) -> None:
    """creates accuracy plot against bin size

    Args:
        accuracies (np.ndarray): each row corresponds to one of 5 splits, each column is the bin size
        model (str): name of the model being analyzed
    """
    bins = [5, 10, 15, 20]
    trace1 = go.Scatter(x=bins, y=accuracies[:, 0], mode="lines", name="split 1", line=dict(dash='longdashdot'))
    trace2 = go.Scatter(x=bins, y=accuracies[:, 1], mode="lines", name="split 2", line=dict(dash='dash'))
    trace3 = go.Scatter(x=bins, y=accuracies[:, 2], mode="lines", name="split 3", line=dict(dash='dashdot'))
    trace4 = go.Scatter(x=bins, y=accuracies[:, 3], mode="lines", name="split 4", line=dict(dash='longdash'))
    trace5 = go.Scatter(x=bins, y=accuracies[:, 4], mode="lines", name="split 5", line=dict(dash='dot'))
    traces = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        title=f"accuracy against bin size for {model}",
        xaxis=dict(title="bin size"),
        yaxis=dict(title="accuracy"),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def F1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """calculate F1 score given true labels and predicted labels

    Args:
        y_true (pd.Series): true labels
        y_pred (pd.Series): predicted labels

    Returns:
        float: f1 score [0-1] 1 being a perfect score
    """
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def F1_plot(F1_scores: np.ndarray, model: str) -> None:
    """creates F1 score plot against bin size

    Args:
        F1_scores (np.ndarray): each row corresponds to one of 5 splits, each column is the bin size
        model (str): name of the model being analyzed
    """
    bins = [5, 10, 15, 20]
    trace1 = go.Scatter(x=bins, y=F1_scores[:, 0], mode="lines", name="split 1")
    trace2 = go.Scatter(x=bins, y=F1_scores[:, 1], mode="lines", name="split 2")
    trace3 = go.Scatter(x=bins, y=F1_scores[:, 2], mode="lines", name="split 3")
    trace4 = go.Scatter(x=bins, y=F1_scores[:, 3], mode="lines", name="split 4")
    trace5 = go.Scatter(x=bins, y=F1_scores[:, 4], mode="lines", name="split 5")
    traces = [trace1, trace2, trace3, trace4, trace5]

    layout = go.Layout(
        title=f"F1 against bin size for {model}",
        xaxis=dict(title="bin size"),
        yaxis=dict(title="F1"),
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    pass


def ROC_plot():
    # for classifiers that only give labels the "curve" will only be one point
    pass
