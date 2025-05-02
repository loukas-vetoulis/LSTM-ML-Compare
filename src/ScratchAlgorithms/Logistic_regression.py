import sys
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.011, epochs=100, lambda_=0.001, decrease_rate=0.1):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.decrease_rate = decrease_rate

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) >= 0.5).astype(int)

    def sigma_sum(self, xi):
        return np.dot(xi, self.weights)

    def compute_z(self, xi):
        return self.sigma_sum(xi) + self.bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def regularization_L2(self, gradient_w):
        gradient_w -= self.lambda_ * self.weights
        return gradient_w

    def fit(self, X, Y):
        reviews, review_vocabulary = X.shape
        self.weights = np.zeros(review_vocabulary)
        self.bias = 0
        lr_temp = self.lr
        
        for epoch in range(self.epochs):
            # Decrease learning rate over time
            current_lr = lr_temp / (1 + self.decrease_rate * epoch)
            
            # Stochastic Gradient Descent
            for i in range(reviews):
                xi = X[i]
                yi = Y[i]
                
                # Forward pass
                z = self.compute_z(xi)
                prediction = self.sigmoid(z)
                
                # Compute gradients
                gradient_w = xi * (yi - prediction)
                gradient_b = yi - prediction
                
                # Apply regularization
                gradient_w = self.regularization_L2(gradient_w)
                
                # Update parameters
                self.weights += current_lr * gradient_w
                self.bias += current_lr * gradient_b
        
        return self
