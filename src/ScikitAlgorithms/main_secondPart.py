from sklearn.linear_model import SGDClassifier
from utils import information_gain_calculation, vocab_cutter 
from utils import information_gain_cutting as info_cut
from utils import downloadData
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
import os
import numpy as np
"""
this code is to run logistic regression and naive bayes and print the results for the best hyperparameters for scikit-learn 
"""
def mainNaiveBayesScikit(n,k):
    x_train,x_test,y_train,y_test=downloadData()
    
    vocabulary = vocab_cutter(n, k,x_train)

    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_test_binary = info_cut(info_gain,6000,x_train,x_test)

    nb = BernoulliNB()
    nb.set_params(alpha=0.9, binarize=0.0)
    nb.fit(X_train_binary, y_train)
    print("Naive Bayes Scikit")
    print(classification_report(y_test, nb.predict(X_test_binary), target_names=["Negative", "Positive"], digits=4))

def mainLogisticRegressionScikit(n,k):
    x_train,x_test,y_train,y_test=downloadData()
    
    vocabulary = vocab_cutter(n, k,x_train)

    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_test_binary = info_cut(info_gain,6000,x_train,x_test)

    logistic_reg = SGDClassifier(
    loss='log_loss',           # For logistic regression
    penalty='l2',              # L2 regularization
    alpha=0.001,               # Equivalent to your lambda_
    learning_rate='invscaling',# Decreasing learning rate
    eta0=0.01,                 # Initial learning rate
    power_t=0.1,              # decrease_rate
    max_iter=1000,             # Number of epochs
    )

    logistic_reg.fit(X_train_binary, y_train)
    print("Logistic Regression Scikit")
    print(classification_report(y_test, logistic_reg.predict(X_test_binary), target_names=["Negative", "Positive"], digits=4))

if __name__ == '__main__':
    mainNaiveBayesScikit(20,20)
    mainLogisticRegressionScikit(20,20)
