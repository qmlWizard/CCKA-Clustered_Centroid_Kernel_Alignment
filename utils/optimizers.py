import numpy as np

def spsa_optimizer(grad_fn, x0, a=0.1, c=0.1, alpha=0.602, gamma=0.101, max_iter=100, callback=None):
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer.

    Args:
        grad_fn: Function to compute loss given parameters.
        x0: Initial parameter vector (numpy array).
        a: Learning rate coefficient.
        c: Perturbation coefficient.
        alpha: Learning rate decay exponent.
        gamma: Perturbation decay exponent.
        max_iter: Number of iterations.
        callback: Optional function called after each iteration with current x.

    Returns:
        x: Optimized parameter vector.
        losses: List of loss values per iteration.
    """
    x = x0.copy()
    losses = []
    for k in range(1, max_iter + 1):
        ak = a / (k ** alpha)
        ck = c / (k ** gamma)
        delta = 2 * np.random.randint(0, 2, size=x.shape) - 1  # Rademacher distribution
        x_plus = x + ck * delta
        x_minus = x - ck * delta
        loss_plus = grad_fn(x_plus)
        loss_minus = grad_fn(x_minus)
        gk = (loss_plus - loss_minus) / (2.0 * ck * delta)
        x = x - ak * gk
        losses.append(grad_fn(x))
        if callback is not None:
            callback(x)
    return x, losses

def parameter_shift_rule(fn, x, shift=np.pi/2):
    """
    Computes the gradient of fn at x using the parameter-shift rule.

    Args:
        fn: Function to evaluate, expects a numpy array input.
        x: Point at which to compute the gradient (numpy array).
        shift: Shift amount (default: pi/2).

    Returns:
        grad: Gradient vector (numpy array).
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += shift
        x_backward[i] -= shift
        grad[i] = 0.5 * (fn(x_forward) - fn(x_backward))
    return grad