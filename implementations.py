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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent; y must be 0 or 1."""
    # TODO: Initialize weights.
    # TODO: Compute probabilities, gradient, and weight updates.
    # TODO: Compute the final loss.
    # return w, loss
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with L2 penalty lambda_ * ||w||²."""
    # TODO: Initialize weights.
    # TODO: Compute the regularized gradient and update weights.
    # TODO: Compute the loss required by your assignment.
    # return w, loss
    raise NotImplementedError