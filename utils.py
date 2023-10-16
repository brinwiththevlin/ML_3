import pandas as pd
import math


def IG(data: pd.DataFrame, target: pd.DataFrame, attr: str) -> float:
    """computes information gain on an attribute

    Args:
        data (pd.DataFrame): dataset
        target (pd.DataFrame): target variable
        attr (str): attribute name

    Returns:
        float: information gain on attr
    """
    global_entropy = H(data, target)
    mysum = 0
    for v in data[attr].values:
        subset = data[data[attr] == v]
        subtarget = target.loc[data[data[attr] == v].index]
        w = len(subset) / len(data)
        subset_entropy = H(subset, subtarget)
        mysum += w * subset_entropy
    return global_entropy - mysum


def H(data: pd.DataFrame, target: pd.DataFrame) -> float:
    """calcluates entorpy

    Args:
        data (pd.DataFrame): data
        target (pd.DataFrame): target variable

    Returns:
        float: entropy
    """
    p0 = target["target"].value_counts()[0]
    p1 = target["target"].value_counts()[1]
    return -(p0 * math.log2(p0) + p1 * math.log2(p1))
