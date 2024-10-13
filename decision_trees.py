import pandas as pd
from typing import Set
from utils import IG


class treeNode:
    def __init__(self, attribute=None, classification=None, branches=None):
        self.attribute = attribute  # The decision attribute for this node
        self.classification = (
            classification  # The classification label if it's a leaf node
        )
        self.branches = (
            branches if branches is not None else {}
        )  # Dictionary to store child nodes
        self.default_classification = None

    def add_branch(self, value, child_node):
        self.branches[value] = child_node

    def is_leaf(self):
        return not bool(self.branches)

    def predict(self, x: pd.Series):
        if self.is_leaf():
            return self.classification
        try:
            return self.branches[x[self.attribute]].predict(x)
        except:
            return self.default_classification


def id3(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, attributes: Set[str]) -> treeNode:
    """generates a decision tree for binary classification

    Args:
        Xtrain (pd.DataFrame): training data
        Ytrain (pd.DataFrame): training lables
        attributes (Set[str]): list of attribute names

    Returns:
        treeNode: decision tree for binary classification
    """
    root = treeNode()
    root.default_classification = root.classification = Ytrain.value_counts().idxmax()

    if len(Ytrain.unique()) == 1:
        root.classification = Ytrain.unique()[0]
        return root
    if len(attributes) == 0:
        root.classification = root.default_classification

    gains = {a: IG(Xtrain, Ytrain, a) for a in attributes}
    A = max(gains)  # A is attribute with greatest gain
    root.attribute = A

    for v in Xtrain[A].unique():
        Xtrain_v = Xtrain[Xtrain[A] == v]
        Ytrain_v = Ytrain.loc[Xtrain[Xtrain[A] == v].index]

        if len(Xtrain_v) == 0:
            common = Ytrain.value_counts().idxmax()
            root.add_branch(v, treeNode(classification=common))
        else:
            root.add_branch(v, id3(Xtrain_v, Ytrain_v, attributes - set([A])))

    return root
