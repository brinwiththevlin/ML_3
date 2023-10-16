import pandas as pd
from typing import List
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

    def add_branch(self, value, child_node):
        self.branches[value] = child_node

    def is_leaf(self):
        return not bool(self.branches)


def id3(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, attributes: List[str]) -> treeNode:
    root = treeNode()

    if len(Ytrain.values) == 1:
        root.classification = Ytrain.values[0]
        return root
    if len(attributes) == 0:
        root.classification = Ytrain.value_counts().idxmax()

    gains = [IG(Xtrain, Ytrain, a) for a in attributes]
    A = attributes[gains.index(max(gains))]  # A is attribute with greatest gain
    root.attribute = A

    for v in Xtrain[A].values:
        Xtrain_v = Xtrain[Xtrain[A] == v]
        Ytrain_v = Ytrain.loc[Xtrain[Xtrain[A] == v].index]

        if len(Xtrain_v) == 0:
            common = Ytrain.value_counts().idxmax()
            root.add_branch(v, treeNode(classification=common))
        else:
            root.add_branch(v, id3(Xtrain_v, Ytrain_v, attributes.remove(A)))

    return root
