import os
import numpy as np
from src.ScratchAlgorithms.Logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from utils import information_gain_cutting as info_cut
from src.ScratchAlgorithms.NaiveBayes import NaiveBayesBinary as naive_bayes

def max_accuracyLogisticRegression(info_gain,m_min,m_max,step,x_train,x_test,y_train,y_test):

    max_accuracy=0
    lambda_dict={"lambda_min":1,"lambda_max":100,"step":10}
    lr_dict={"lr_min":1,"lr_max":100,"step":10}
    
    max_report, max_lr, max_lambda, max_m = None, None, None, None
    
    for i in range(m_min,m_max,step):
        
        print("Started info cut: ")

        # X_train_binary,X_test_binary = info_cut(info_gain,i,x_train,x_test)
        X_train_binary = np.load(os.path.join('data', 'X_train_binary.npy'))
        X_test_binary = np.load(os.path.join('data', 'X_test_binary.npy'))
        print("Finished info cut: ")

        for j in range(lambda_dict["lambda_min"],lambda_dict["lambda_max"],lambda_dict["step"]):

            for k in range(lr_dict["lr_min"],lr_dict["lr_max"],lr_dict["step"]):
                # Now you can use these variables as your data in the model
                lg = LogisticRegression(lr=k/1000,epochs=100,lambda_=j/1000)
                lg.fit(X_train_binary, y_train)
                Y_pred = lg.predict(X_test_binary)

                accuracy = accuracy_score(y_test, Y_pred)
                report = classification_report(y_test, Y_pred, target_names=["Negative", "Positive"], digits=4)

                if accuracy>max_accuracy:
                    max_accuracy=accuracy
                    max_m=i
                    max_report=report
                    max_lambda=j/1000
                    max_lr=k/1000
                    print(max_report)
                    print(f"Lambda: {max_lambda}, LR: {max_lr}")

    return max_accuracy,max_m,max_report,max_lambda,max_lr

def max_accuracyNaiveBayes(info_gain,x_train,y_train,x_test,y_test,m_min,m_max,step):
    max_accuracy=0
    max_report,max_laplace,max_m=None,None,None

    laplace_dict={"laplace_min":1,"laplace_max":50,"step":1}

    for i in range(m_min,m_max,step):

        print("Started info cut: ")

        X_train_binary,X_test_binary = info_cut(info_gain,i,x_train,x_test)

        print("Finished info cut: ")

        for j in range(laplace_dict["laplace_min"],laplace_dict["laplace_max"],laplace_dict["step"]):
            naive_bayes_binary = naive_bayes(j/10)
            naive_bayes_binary.train(X_train_binary, y_train)
            test_accuracy = naive_bayes_binary.accuracy(X_test_binary, y_test)
            report = classification_report(y_test, naive_bayes_binary.predict(X_test_binary), target_names=["Negative", "Positive"], digits=4)
            
            if test_accuracy>max_accuracy:
                max_accuracy=test_accuracy
                max_m=i
                max_laplace=j
                max_report=report
                print(max_report)
                print(f"Laplace: {j/10}")

    return max_accuracy,max_m,max_laplace/10,max_report