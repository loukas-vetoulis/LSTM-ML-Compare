from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from scipy.stats import entropy
from sklearn.naive_bayes import BernoulliNB
from utils import *
from machine_learning import *

"""
this code is to calculate the best hyperparameters for logistic regression and naive bayes and print the results
"""


def mainLogisticRegression(n,k,m_min,m_max,step):

    x_train,x_test,y_train,y_test=downloadData()

    # vocabulary = vocab_cutter(n, k,x_train)

    # info_gain = information_gain_calculation(vocabulary,x_train,y_train)
    info_gain = np.load(os.path.join("data","info_gain.npy"),allow_pickle=True).item()
    max_accur,max_m,string,max_lambda,max_lr=max_accuracyLogisticRegression(info_gain,m_min,m_max,step,x_train,x_test,y_train,y_test)
    
    print(string)
    print(f"max accuracy: {max_accur}, max m: {max_m}, max lambda: {max_lambda}, max lr: {max_lr}")

def mainNaiveBayes(n,k,m_min,m_max,step):

    x_train,x_test,y_train,y_test=downloadData()
    
    vocabulary = vocab_cutter(n, k,x_train)

    info_gain = information_gain_calculation(vocabulary,x_train,y_train)

    max_accur,max_m,max_laplace,string=max_accuracyNaiveBayes(info_gain,x_train,y_train,x_test,y_test,m_min,m_max,step)
    
    print(string)
    print(f"max accuracy: {max_accur}, max m: {max_m}, max laplace: {max_laplace}")


if __name__ == '__main__':
    m_dict={"m_min":1000,"m_max":10001,"step":1000}
    mainLogisticRegression(20,20,m_dict["m_min"],m_dict["m_max"],m_dict["step"])
    mainNaiveBayes(20,20,m_dict["m_min"],m_dict["m_max"],m_dict["step"])
    