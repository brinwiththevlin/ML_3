import decision_trees as dt
import naive_bayes as nb
import analysis
from utils import discretize
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random


if __name__ == "__main__":
    random.seed(50)
    data = datasets.load_iris()  # load dataset from sklearn
    # convert features to pandas
    iris_df = pd.DataFrame(data.data, columns=data.feature_names)
    # set setosa as 1 and the other two as 0
    iris_df["target"] = (data.target == 0).astype(int)

    bins = [5, 10, 15, 20]
    accuracies = np.zeros(shape=(4, 5))
    F1_score = np.zeros(shape=(4, 5))

    states = [random.randint(10, 50) for __ in range(5)]
    ######################################
    # ------------ binning  ------------ #
    ######################################
    binned_data = [discretize(iris_df, bin) for bin in bins]

    ######################################
    # ---------- ID3 training ---------- #
    ######################################
    for data in binned_data:
        for i in range(5):
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                data.drop(["target"]),
                data["target"],
                test_size=0.33,
                random_state=states[i],
            )

            model = dt.id3(Xtrain, Ytrain)
