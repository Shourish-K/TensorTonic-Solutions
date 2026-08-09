# -----------------------------------------------------------------------------
# Import Libraries
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    N = X.shape[0]
    
    # Initialize weights and biases to be 0
    
    w = np.zeros(X.shape[1])
    b = 0
    
    # Training loop for n = steps
    
    for step in range(steps):
        logit = X @ w + b
        y_hat = _sigmoid(logit)
        w += lr * ((1/N) * (X.T @ (y - y_hat)))
        b += lr * (np.mean(y - y_hat))
    
    
    return (w,b)
    