import pandas as pd
import json

def get_feature_probs(data, Xtrain, attributes, bins):
    class0_probs = dict()
    class1_probs = dict()
    alpha = 100
    train_len = Xtrain.shape[0]
    
    print(train_len)
    for a in attributes:
        
        x_vals = Xtrain[a].value_counts().to_dict()
        
        temp0 = dict()
        temp1 = dict()
        for k in x_vals:
            
            #print(temp0)
            temp0[k] = (len(data[(data[a] == k) & (data['target'] == 0)])+alpha)/(train_len + (bins*alpha))
            temp1[k] = (len(data[(data[a] == k) & (data['target'] == 1)])+alpha)/(train_len + (bins*alpha))
        
        
        class0_probs[a] = temp0
        class1_probs[a] = temp1
        
    return class0_probs, class1_probs

def prob_features_given_class(class_probs: pd.DataFrame):
    prob = 1
    for attribtue in class_probs:
        for k in class_probs[attribtue]:
            prob *= class_probs[attribtue][k]
    print(prob)
   
        
        
def naive_bayes( Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, attributes, bins:int):
    train_len = Ytrain.shape[0]
    value_counts = Ytrain.value_counts()
    p_0 = value_counts[0]/train_len
    p_1 = value_counts[1]/train_len
    
    data = pd.concat([Xtrain, Ytrain], axis=1)
    f = open("test.txt", "w")
    f.write(str(data.sort_values(by='sepal length (cm)')))
    class_0_probs, class_1_probs = get_feature_probs(data=data,Xtrain=Xtrain, attributes=attributes, bins=bins)
    #print(class_1_probs)
    prob_features_given_class(class_0_probs)
    