import numpy as np

def calculate_primal_objective(y, X, w, lambda_):
    """Mean hinge loss plus L2 regularization, including the intercept."""
    hinge_loss = np.maximum(1 - y * (X @ w), 0)
    return float(np.mean(hinge_loss) + lambda_ / 2 * np.sum(w**2))


def calculate_stochastic_gradient(y, X, w, lambda_, n):
    """Regularized hinge-loss subgradient for row n."""
    gradient = lambda_ * w
    if y[n] * (X[n] @ w) < 1:
        gradient = gradient - y[n] * X[n]
    return gradient


def calculate_coordinate_update(y, X, lambda_, alpha, w, n):
    """Update one dual coordinate and the corresponding weights in place."""
    x_n, y_n = X[n], y[n]
    squared_norm = x_n @ x_n
    old_alpha = alpha[n]
    if squared_norm == 0:
        # Zero feature rows contribute constant hinge loss and no weight update.
        alpha[n] = 1.0
    else:
        scale = lambda_ * len(y)
        margin_gap = 1 - y_n * (x_n @ w)
        alpha[n] = np.clip(old_alpha + scale * margin_gap / squared_norm, 0.0, 1.0)
        w += 1.0 / scale * (alpha[n] - old_alpha) * y_n * x_n
    return w, alpha


def calculate_dual_objective(w, alpha, lambda_):
    """Dual objective for weights consistent with the dual coordinates."""
    return float((1 / len(alpha)) * np.sum(alpha) - lambda_ / 2 * np.sum(w**2))
