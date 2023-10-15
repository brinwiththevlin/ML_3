# %%
import decision_trees as dt
import naive_bayes as nb
import analysis
from sklearn import datasets
import pandas as pd

# %%
if __name__ == "__main__":
    data = datasets.load_iris() # load dataset from sklearn
    
    iris_df = pd.DataFrame(data.data, columns=data.feature_names) # convert features to pandas
    iris_df["target"] = (data.target == 0).astype(int) # set setosa as 1 and the other two as 0
    
# %%
