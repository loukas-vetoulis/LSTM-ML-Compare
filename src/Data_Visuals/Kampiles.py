import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def learning_curve(model, func,X_train, y_train, X_develop, y_develop, step=2500):

    train_precisions, train_recalls, train_f1s = [], [], []
    develop_precisions, develop_recalls, develop_f1s = [], [], []
    train_sizes = []


    for i in range(step, len(X_train) + 1, step):
        # Επιλογή υποσυνόλου εκπαίδευσης
        X_train_subset = X_train[:i]
        y_train_subset = y_train[:i]
        

        func(X_train_subset, y_train_subset)

        train_predict = model.predict(X_train_subset)
        develop_predict = model.predict(X_develop)


        train_precisions.append(precision_score(y_train_subset, train_predict))
        train_recalls.append(recall_score(y_train_subset, train_predict))
        train_f1s.append(f1_score(y_train_subset, train_predict))


        develop_precisions.append(precision_score(y_develop, develop_predict))
        develop_recalls.append(recall_score(y_develop, develop_predict))
        develop_f1s.append(f1_score(y_develop, develop_predict))


        train_sizes.append(i)


    plt.figure(figsize=(10, 6))


    plt.plot(train_sizes, train_precisions, label='Train Precision', color='blue')
    plt.plot(train_sizes, develop_precisions, label='Develop Precision', color='blue', linestyle='--')

    plt.plot(train_sizes, train_recalls, label='Train Recall', color='green')
    plt.plot(train_sizes, develop_recalls, label='Develop Recall', color='green', linestyle='--')

    plt.plot(train_sizes, train_f1s, label='Train F1', color='red')
    plt.plot(train_sizes, develop_f1s, label='Develop F1', color='red', linestyle='--')

    # Ρυθμίσεις του γραφήματος
    plt.title('Learning Curves')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()
