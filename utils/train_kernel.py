import pennylane as qml
from pennylane import numpy as np


def generate_origin_kernel_matrix(x, kernel):
    km = []
    for point in x:
        distance = kernel(point, np.zeros(point.shape))
        km.append(distance)

    return np.asarray(km, requires_grad=True).reshape(-1, 1)


def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False, 
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )


    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product


def target_alignment_towards_origin(
    X,
    Y,
    kernel,
    params,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    
    kernel_matrix = generate_origin_kernel_matrix(X, kernel)

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)


    #numerator = np.sum(_Y * np.array(K))
    #denominator = np.sqrt(np.sum(K) * np.sum(_Y**2))


    T = np.outer(_Y, _Y)
    inner_product = np.sum(kernel_matrix * T)
    norm = np.sqrt(np.sum(kernel_matrix * kernel_matrix) * np.sum(T * T))
    inner_product = inner_product / norm
    
    #ta = numerator / denominator

    #kao = ta + 0.1 + np.linalg.norm(params)**2

    #return kao
    return inner_product
