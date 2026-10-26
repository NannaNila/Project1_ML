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
    """Return final weights and half-MSE after full-batch gradient descent."""
    w = initial_w.copy()

    for _ in range(max_iters):
        err = y - tx.dot(w)
        grad = -tx.T.dot(err) / len(err)
        w = w - gamma * grad

    # Evaluate at the returned weights, including when max_iters is zero.
    loss = 0.5 * np.mean((y - tx.dot(w)) ** 2)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    # Initialize weights.

    w = initial_w.copy()

    # Sample a data point and update weights at each iteration.

    for n_iter in range(max_iters):
        start_index = np.random.randint(len(y))
        end_index = start_index + 1
        y_batch, tx_batch = y[start_index:end_index], tx[start_index:end_index]
        err = y_batch - tx_batch.dot(w)
        grad = -tx_batch.T.dot(err) / len(err)
        w = w - gamma * grad

    # Compute the final loss on the full dataset.

    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    # Solve the normal equations for the weights.
    w = np.linalg.solve(a, b)
    # Compute the final loss.
    loss = 0.5 * np.mean((y - tx.dot(w)) ** 2)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    # Construct and solve the regularized normal equations.
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    # Return half-MSE without the regularization penalty.

    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    return w, loss


def sigmoid(z):
    """Compute the sigmoid without overflowing for large negative inputs."""
    return np.exp(-np.logaddexp(0, -z))


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
    loss = np.mean(np.logaddexp(0, z) - y * z)  # logaddexp(a, b) => log(e^a + e^b)
    return w, loss


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
    # Return the mean logistic loss without the regularization penalty.
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return w, loss
