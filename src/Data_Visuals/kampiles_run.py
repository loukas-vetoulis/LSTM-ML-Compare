import os
from Logistic_regression import LogisticRegression
import numpy as np
from NaiveBayes import NaiveBayesBinary    
from utils import downloadData
from sklearn.model_selection import train_test_split
from Kampiles import learning_curve 
from sklearn.naive_bayes import BernoulliNB
from utils import information_gain_calculation, information_gain_cutting as info_cut, vocab_cutter
from sklearn.linear_model import SGDClassifier


def kampiles_run_NaiveBayesFirst(n,k):
    x_train,x_test,y_train,y_test=downloadData()
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.16, random_state=1)

    vocabulary = vocab_cutter(n, k,x_train)
    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_dev_binary = info_cut(info_gain,6000,x_train,x_dev)

    nb = NaiveBayesBinary(0.9) 

    learning_curve(nb, nb.train,X_train_binary, y_train, X_dev_binary, y_dev, step=2500)

def kampiles_run_LogisticRegressionFirstPart(n,k):

    x_train,x_test,y_train,y_test=downloadData()
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.16, random_state=1)

    vocabulary = vocab_cutter(n, k,x_train)
    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_dev_binary = info_cut(info_gain,6000,x_train,x_test)

    lg = LogisticRegression()
    learning_curve(lg, lg.fit, X_train_binary, y_train, X_dev_binary, y_dev, step=2500)

def kampiles_run_NaiveBayesSecond(n,k):
    x_train,x_test,y_train,y_test=downloadData()
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.16, random_state=1)

    vocabulary = vocab_cutter(n, k,x_train)
    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_dev_binary = info_cut(info_gain,6000,x_train,x_test)

    nb = BernoulliNB()
    nb.set_params(alpha=0.9, binarize=0.0)

    learning_curve(nb, nb.fit,X_train_binary, y_train, X_dev_binary, y_dev, step=2500)

def kampiles_run_LogisticRegressionSecondPart(n,k):
    x_train,x_test,y_train,y_test=downloadData()
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.16, random_state=1)

    vocabulary = vocab_cutter(n, k,x_train)
    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_dev_binary = info_cut(info_gain,6000,x_train,x_test)


    logistic_reg = SGDClassifier(
    loss='log_loss',           # For logistic regression
    penalty='l2',              # L2 regularization
    alpha=0.001,               # Equivalent to your lambda_
    learning_rate='invscaling',# Decreasing learning rate
    eta0=0.01,                 # Initial learning rate
    power_t=0.1,              # decrease_rate
    max_iter=1000,             # Number of epochs
    )

    learning_curve(logistic_reg,logistic_reg.fit, X_train_binary, y_train, X_dev_binary, y_dev, step=2500)

if __name__ == '__main__':
    kampiles_run_NaiveBayesFirst(20,20)
    kampiles_run_LogisticRegressionFirstPart(20,20)
    kampiles_run_NaiveBayesSecond(20,20)
    kampiles_run_LogisticRegressionSecondPart(20,20)