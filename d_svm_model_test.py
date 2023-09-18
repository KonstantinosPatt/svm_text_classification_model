# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:05:53 2021

@author: kosti
"""

import pickle
from sklearn.metrics import accuracy_score

# Use the function from train 
from train import create_svm_model

def test_svm_model(file, part, percentile, train_csv):

    test_file, one_hot = create_svm_model(file, part, percentile, train_csv)
    
    # load model
    model = 'model.sav'
    loaded_model = pickle.load(open(model, 'rb'))
    
    data = test_file.dropna()    # Removes NaN values
    temp_data = data.to_numpy()  # Converts the dataframe into a numpy array
    X = temp_data[:,:-1]         # Gets the features
    y = temp_data[:,-1]          # Gets the output values
    y = y.astype('int')          # Defines y as integer

    preds = loaded_model.predict(X)
    accuracy = accuracy_score(y, preds)
    #print("Predicted values:", preds)
    #print("   Actual values:", y)
    #print("Accuracy score:", accuracy)
    return accuracy


if __name__ == '__main__':
    file = "./output"
    train_csv = "train_one_hot.csv"
    percentile = 100

    part = [9]
    test_svm_model(file, part, percentile, train_csv)