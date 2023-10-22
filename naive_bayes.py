import pandas as pd
import json

def predict (Xtest, Ytest, prob_dict):
    c0 = prob_dict["c0"]
    c1 = prob_dict["c1"]
    c0_features= prob_dict["c0_features"]
    c1_features = prob_dict["c1_features"]
    result_list = []
    
    for item in Xtest.iterrows():
        point = item[1].to_dict()
        
        numerator0 = 1
        numerator1 = 1
        
        for attribute in point:
            
            numerator0 *= c0_features[attribute][point[attribute]]
            numerator1 *= c1_features[attribute][point[attribute]]
        numerator1 *= c1
        numerator0 *= c0
        if(numerator0>numerator1):
            result_list.append(0)
        else:
            result_list.append(1)
    Ypred = pd.Series(result_list, index=Ytest.index)
    
    return Ypred
            
    '''
    testx = Xtest.iloc[0].to_dict()
    mult0 = 1
    mult1 = 1
    for a in testx:
        mult0 *= c0_features[a][testx[a]]
        mult1 *= c1_features[a][testx[a]]

    mult0 *= c0
    mult1 *= c1
    
    if(mult0>mult1):
        return 0
    else:
        return 1
    '''

def get_feature_probs(data, Xtrain, attributes, bins):
    class0_probs = dict()
    class1_probs = dict()
    alpha = 10
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
    data = pd.concat([Xtrain, Ytrain], axis=1)
    
    prob_dict = dict()
    p_0 = value_counts[0]/train_len
    p_1 = value_counts[1]/train_len
    prob_dict["c0"] = p_0
    prob_dict["c1"] = p_1
    
    class_0_probs, class_1_probs = get_feature_probs(data=data,Xtrain=Xtrain, attributes=attributes, bins=bins)
    prob_dict["c0_features"] = class_0_probs
    prob_dict["c1_features"] = class_1_probs
    
    
    return prob_dict

    
    