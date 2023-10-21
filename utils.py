import pandas as pd
import numpy as np
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
    p0 = target.value_counts()[0] if 0 in target.value_counts() else 0
    p1 = target.value_counts()[1] if 1 in target.value_counts() else 0
    return -(p0 * math.log2(p0) + p1 * math.log2(p1)) if p0 != 0 and p1 != 0 else 0


def discretize(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    """discretizes all features of df using specified number of bins

    Args:
        df (pd.DataFrame): the original data frame
        bins (int): the number of bins

    Returns:
        pd.DataFrame: the discretized data frame
    """
    for feature in df.columns:
        if feature == "target":
            continue
        edges = np.histogram_bin_edges(df[feature], bins)
        means = {i: (edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)}
        edges[0] -= 0.2
        indexes = np.digitize(df[feature], edges, right=True) - 1
        new_values = [means[i] for i in indexes]
        df[feature] = new_values

    return df
