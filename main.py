import decision_trees as dt
import naive_bayes as nb
import analysis
from utils import discretize
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from typing import Tuple


if __name__ == "__main__":
    random.seed(50)
    data = datasets.load_iris()  # load dataset from sklearn
    # convert features to pandas
    iris_df = pd.DataFrame(data.data, columns=data.feature_names)
    # set setosa as 1 and the other two as 0
    iris_df["target"] = (data.target == 0).astype(int)

    bins = [5, 10, 15, 20]
    accuracies = np.zeros(shape=(4, 5), dtype=float)
    F1_scores = np.zeros(shape=(4, 5), dtype=float)
    ROC_scores = np.zeros(shape=(4, 5), dtype=tuple)

    states = [random.randint(10, 100) for __ in range(5)]
    ######################################
    # ------------ binning  ------------ #
    ######################################
    binned_data = [discretize(iris_df.copy(), bin) for bin in bins]

    ######################################
    # ---------- ID3 training ---------- #
    ######################################
    
    '''
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                binned_data[0].drop(columns=["target"]),
                binned_data[0]["target"],
                test_size=0.33,
                random_state=states[1],
            )
    
    probs = nb.naive_bayes(Xtrain=Xtrain, Ytrain=Ytrain,attributes=list(Xtrain.columns), bins =5)
    Ypred = nb.predict(Xtest=Xtest, Ytest=Ytest, prob_dict=probs)
    accuracy = analysis.accuracy(y_true=Ytest, y_pred=Ypred)
    f1 = analysis.F1(y_pred=Ypred, y_true=Ytest)
    print(f1)
    
'''
    for i, data in enumerate(binned_data):
    
        for j in range(5):
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                data.drop(columns=["target"]),
                data["target"],
                test_size=0.33,
                random_state=states[j],
            )
           

            model = dt.id3(Xtrain, Ytrain, set(Xtrain.columns))
            Ypred = pd.Series(
                [model.predict(x) for _, x in Xtest.iterrows()], index=Ytest.index
            )
            accuracies[i, j] = analysis.accuracy(Ytest, Ypred)
            F1_scores[i, j] = analysis.F1(Ytest, Ypred)
            ROC_scores[i, j] = analysis.ROC(Ytest, Ypred)

    analysis.print_accuracy(accuracies, bins=bins)
    analysis.accuracy_plot(accuracies=accuracies, model="ID3")
    analysis.F1_plot(F1_scores=F1_scores, model="ID3")
    analysis.ROC_plot(ROC_scores=ROC_scores, model="ID3")
