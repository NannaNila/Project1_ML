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


def calculate_squared_primal_objective(y, X, w, lambda_, sample_weights=None):
    """Half mean squared hinge plus lambda/2 * ||w||², including intercept."""
    margins = np.maximum(1 - y * (X @ w), 0)
    squared = margins**2
    if sample_weights is not None:
        squared = sample_weights * squared
    return float(0.5 * np.mean(squared) + lambda_ / 2 * (w @ w))


def calculate_squared_stochastic_gradient(y, X, w, lambda_, n, sample_weights=None):
    margin = max(0.0, 1 - y[n] * (X[n] @ w))
    weight = 1 if sample_weights is None else sample_weights[n]
    return lambda_ * w - weight * y[n] * X[n] * margin


def calculate_squared_coordinate_update(y, X, lambda_, alpha, w, n, sample_weights=None):
    """Exact coordinate ascent: squared-hinge dual has alpha >= 0, no upper cap.

    Dual: mean(alpha - alpha²/(2*c)) - lambda/2 * ||w||²,
    where c is the sample weight (one by default),
    with w = X.T @ (y * alpha) / (lambda * N).
    """
    x_n, y_n = X[n], y[n]
    scale = lambda_ * len(y)
    old_alpha = alpha[n]
    inverse_weight = 1 if sample_weights is None else 1 / sample_weights[n]
    derivative = 1 - y_n * (x_n @ w) - old_alpha * inverse_weight
    curvature = inverse_weight + (x_n @ x_n) / scale
    alpha[n] = max(0.0, old_alpha + derivative / curvature)
    w += (alpha[n] - old_alpha) / scale * y_n * x_n
    return w, alpha


def calculate_squared_dual_objective(w, alpha, lambda_, sample_weights=None):
    quadratic = alpha**2 if sample_weights is None else alpha**2 / sample_weights
    return float(np.mean(alpha - 0.5 * quadratic) - lambda_ / 2 * (w @ w))
