import numpy as np

# Shared parameters:
# y         : target values, shape (N,)
# tx        : input matrix, shape (N, D)
# initial_w : initial weights, shape (D,)
# max_iters : number of optimization iterations
# gamma     : learning rate
# lambda_   : regularization strength
#
# Returns:
# w         : learned weights, shape (D,)
# loss      : final loss (scalar)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    # TODO: Initialize weights.
    # TODO: Compute the gradient and update weights at each iteration.
    # TODO: Compute the final loss.
    # return w, loss
    raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    # TODO: Initialize weights.
    # TODO: Sample a data point and update weights at each iteration.
    # TODO: Compute the final loss on the full dataset.
    # return w, loss
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    # TODO: Solve the normal equations for the weights.
    # TODO: Compute the final loss.
    # return w, loss
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    # TODO: Construct and solve the regularized normal equations.
    # TODO: Compute the loss required by your assignment.
    # return w, loss
    raise NotImplementedError

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent; y must be 0 or 1."""
    # Initialize weights.
    w = initial_w.copy()
    n = len(y)

    for i in range(max_iters):
        p = sigmoid(tx @ w)
        error = p - y
        gradient = (tx.T @ error) / n
        w = w - gamma * gradient

    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z) # logaddexp(a, b) => log(e^a + e^b)
    return z, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with L2 penalty lambda_ * ||w||²."""
    # Initialize weights.
    w = initial_w.copy()
    n = len(y)

    for i in range(max_iters):
        p = sigmoid(tx @ w)
        error = p - y
        gradient = (tx.T @ error) / n + 2 * lambda_ * w        
        w = w - gamma * gradient

    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z) + lambda_ * np.sum(w**2) # this is the term penalizing large weights to reduce overfitting
    return z, loss