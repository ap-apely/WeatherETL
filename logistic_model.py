"""
Logistic regression model
"""
import numpy as np

class LogisticRegression:
    """
    Logistic regression model
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize model parameters
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None

    def sigmoid(self, z):
        """
        Compute sigmoid function
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def initialize_weights(self, dim):
        """
        Initialize weights and bias
        """
        w = np.random.randn(dim, 1) * 0.01
        b = 0
        return w, b

    def hypothesis(self, w, X, b):
        """
        Compute hypothesis
        """
        z = np.dot(w.T, X) + b
        return self.sigmoid(z)

    def compute_cost(self, A, Y):
        """
        Compute cost
        """
        m = Y.shape[1]
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return np.squeeze(cost)

    def compute_gradients(self, X, Y, A):
        """
        Compute gradients
        """
        m = X.shape[1]
        dw = (1/m) * np.dot(X, (A - Y).T)
        db = (1/m) * np.sum(A - Y)
        return dw, db

    def train_model(self, X_train, Y_train, X_test=None, Y_test=None, print_cost=False):
        """
        Train model
        """
        costs = []
        train_accuracies = []
        test_accuracies = []
        
        self.w, self.b = self.initialize_weights(X_train.shape[0])
        
        for i in range(self.num_iterations):
            A = self.hypothesis(self.w, X_train, self.b)
            
            if i % 100 == 0:
                # Compute cost
                cost = self.compute_cost(A, Y_train)
                costs.append(cost)
                
                # Compute accuracies
                train_accuracy = self.compute_accuracy(X_train, Y_train)
                train_accuracies.append(train_accuracy)
                
                if X_test is not None and Y_test is not None:
                    test_accuracy = self.compute_accuracy(X_test, Y_test)
                    test_accuracies.append(test_accuracy)
                
                if print_cost:
                    print(f"Cost after iteration {i}: {cost:.6f}")
            
            dw, db = self.compute_gradients(X_train, Y_train, A)
            
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        
        # Calculate final accuracies
        train_accuracy = self.compute_accuracy(X_train, Y_train)
        test_accuracy = self.compute_accuracy(X_test, Y_test) if X_test is not None and Y_test is not None else None
        
        return {
            "w": self.w,
            "b": self.b,
            "costs": costs,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }

    def predict(self, X):
        """
        Make predictions
        """
        A = self.hypothesis(self.w, X, self.b)
        return (A > 0.5).astype(int)

    def compute_accuracy(self, X, Y):
        """
        Compute accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == Y) * 100
