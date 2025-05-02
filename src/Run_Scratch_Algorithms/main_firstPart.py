from src.lib.utils import information_gain_calculation, vocab_cutter
from src.lib.utils import information_gain_cutting as info_cut
from src.lib.utils import downloadData
from src.ScratchAlgorithms.Logistic_regression import LogisticRegression
from src.ScratchAlgorithms.NaiveBayes import *
from sklearn.metrics import classification_report

"""
this code is to run logistic regression and naive bayes and print the results for the best hyperparameters for our algorithms
"""
def mainNaiveBayes(n,k):
    x_train,x_test,y_train,y_test=downloadData()
    
    vocabulary = vocab_cutter(n, k,x_train)

    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_test_binary = info_cut(info_gain,6000,x_train,x_test)

    nb = NaiveBayesBinary(0.9)
    nb.train(X_train_binary, y_train)

    print("Naive Bayes Scikit")
    print(classification_report(y_test, nb.predict(X_test_binary), target_names=["Negative", "Positive"], digits=4))

def mainLogisticRegression(n,k):
    x_train,x_test,y_train,y_test=downloadData()
    
    vocabulary = vocab_cutter(n, k,x_train)

    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    X_train_binary,X_test_binary = info_cut(info_gain,6000,x_train,x_test)

    lg = LogisticRegression()
    lg.fit(X_train_binary, y_train)

    print("Logistic Regression Scikit")
    print(classification_report(y_test, lg.predict(X_test_binary), target_names=["Negative", "Positive"], digits=4))

if __name__ == '__main__':
    mainNaiveBayes(20,20)
    mainLogisticRegression(20,20)
