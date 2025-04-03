import torch
from mpmath import mp


def eye_init(M: int) -> torch.FloatTensor:
    """
    Generates an identity matrix of size M x M.

    :param M: Number of units.

    :return: Identity matrix.
    """

    if M <= 0:
        raise ValueError(f"Invalid number of units <<{M}>>. Must be positive.")

    return torch.eye(M, dtype=torch.float32, requires_grad=False)


def sparse_tensor_init(M: int, N: int, C: int = 1, distribution: str = 'uniform', scaling: float = 1.0,
                       signs_from: str | None = None) -> torch.FloatTensor:
    """
    Generates an M x N matrix to be used as sparse kernel. For each row only C elements are non-zero (i.e.,
    each input dimension is projected only to C neurons). The non-zero elements are generated randomly from a uniform
    distribution in [-1,1], from a standard normal distribution scaled to a variance equal to 1/C,
    or they are generated deterministically all equals to 1 if distribution is 'fixed'.
    In the latter case, the signs can be generated from different sources.
    They are generated based on the digits of pi, e, a logistic map or randomly.
    In particular, for pi and e, the signs are thresholded at 4.5.
    For the logistic map, the signs are generated based on the value of the logistic map and a threshold of 0.5.

    :param M: Number of rows.
    :param N: Number of columns.
    :param C: Number of nonzero elements.
    :param distribution: Initialisation strategy. It can be 'uniform', 'normal' or 'fixed'.
    :param scaling: Scaling factor for fixed and uniform distribution.
    :param signs_from: The source of the signs for the fixed distribution.

    :return: M x N matrix.
    """

    if C > N:
        raise ValueError(f"Invalid number of non-zero elements <<{C}>>. Must be less than or equal to N.")
    if M <= 0 or N <= 0:
        raise ValueError(f"Invalid number of units <<{M}>> and <<{N}>>. Must be positive.")
    if C <= 0:
        raise ValueError(f"Invalid number of non-zero elements <<{C}>>. Must be positive.")
    if distribution not in ['uniform', 'normal', 'fixed']:
        raise ValueError(f"Invalid distribution <<{distribution}>>. Only uniform, normal and fixed allowed.")
    if signs_from not in [None, 'random', 'pi', 'e', 'logistic']:
        raise ValueError(f"Invalid signs_from <<{signs_from}>>. Only random, pi, e and logistic allowed.")

    if distribution == 'fixed':
        values = torch.ones(M * C, dtype=torch.float32) * scaling

        if signs_from is None or signs_from == 'random':
            # Generate random signs
            signs = torch.randint(0, 2, (M * C,)).mul(2).sub(1)
        elif signs_from == 'pi':
            # Set precision to get enough decimal digits
            mp.dps = M * C + 2

            digits = str(mp.pi)[2:2 + M * C]
            signs = torch.tensor([1 if int(d) > 4.5 else -1 for d in digits], dtype=torch.float32)
        elif signs_from == 'e':
            # Set precision to get enough decimal digits
            mp.dps = M * C + 2

            digits = str(mp.e)[2:2 + M * C]
            signs = torch.tensor([1 if int(d) > 4.5 else -1 for d in digits], dtype=torch.float32)
        elif signs_from == 'logistic':
            # Logistic map initialization
            x = 0.33
            signs = [0] * M * C
            for i in range(M * C):
                x = 4 * x * (1 - x)
                signs[i] = 1 if x > 0.5 else -1
            signs = torch.tensor(signs, dtype=torch.float32)

        values *= signs
    elif distribution == 'uniform':
        values = (torch.rand(M * C, dtype=torch.float32) * 2 - 1) * scaling
    elif distribution == 'normal':
        values = torch.randn(M * C, dtype=torch.float32) / torch.sqrt(torch.tensor(C, dtype=torch.float32))

    indices = torch.zeros((M * C, 2), dtype=torch.long)
    for i in range(M):
        idx = torch.randperm(N)[:C]
        indices[i * C:(i + 1) * C, 0] = i
        indices[i * C:(i + 1) * C, 1] = idx

    return torch.sparse_coo_tensor(indices.T, values, (M, N), dtype=torch.float32, requires_grad=False).to_dense()


def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """
    Rescales W to have rho(W) = rho_desired

    :param W: Weight matrix.
    :param rho_desired: Desired spectral radius.

    :return: W rescaled to have spectral radius equal to rho_desired.
    """

    e = torch.linalg.eigvals(W)
    rho_curr = torch.max(torch.abs(e)).item()
    return W * (rho_desired / rho_curr)


def fast_spectral_rescaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """
    Rescales a W uniformly sampled in (-1,1) to have rho(W) = rho_desired.
    This method is fast since we don't need to compute the spectrum of W, which is very slow.

    NB: this method works only if W is uniformly sampled in (-1,1).
    In particular, W must be fully connected!

    :param W: There must be a square matrix uniformly sampled in (-1,1), fully connected.
    :param rho_desired: Desired spectral radius.

    :return: W rescaled to have spectral radius equal to rho_desired.
    """

    units = W.shape[0]
    value = ((rho_desired / torch.sqrt(torch.tensor(units, dtype=torch.float32)))
             * (6 / torch.sqrt(torch.tensor(12, dtype=torch.float32))))
    return W * value


def circular_tensor_init(M: int, distribution: str = 'fixed', scaling: float = 1.0) -> torch.FloatTensor:
    """
    Generates an M x M matrix with ring topology.
    Each neuron receives input only from one neuron and propagates its activation only to one other neuron.
    The non-zero elements are in positions (i+1, i) and (0, M-1).
    The non-zero elements are generated randomly from a uniform distribution in [-1,1],
    from a standard normal distribution or are fixed to 1.

    :param M: Number of hidden units
    :param distribution: Initialisation strategies. It can be 'uniform', 'normal' or 'fixed'.
    :param scaling: Scaling factor for fixed and uniform distribution.

    :return: M x M sparse matrix.
    """

    if M <= 0:
        raise ValueError(f"Invalid number of units <<{M}>>. Must be positive.")
    if distribution not in ['uniform', 'normal', 'fixed']:
        raise ValueError(f"Invalid distribution <<{distribution}>>. Only uniform, normal and fixed allowed.")

    dense_shape = (M, M)
    # Generate the indices of the non-zero elements, (i+1, i) and (0, M-1)
    indices = torch.cat([torch.stack([torch.arange(1, M), torch.arange(M - 1)], dim=1), torch.tensor([[0, M - 1]])],
                        dim=0)

    if distribution == 'fixed':
        values = torch.ones(M, dtype=torch.float32) * scaling
    elif distribution == 'uniform':
        values = (torch.rand(M, dtype=torch.float32) * 2 - 1) * scaling
    elif distribution == 'normal':
        values = torch.randn(M, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices.T, values, dense_shape, dtype=torch.float32, requires_grad=False).to_dense()


def skewsymmetric(M: int, scaling: float = 1.0) -> torch.FloatTensor:
    """
    Generate a skewsymmetric matrix, i.e., a matrix A such that A^T = -A.
    The entries of the matrix are generated randomly from a uniform distribution in [-1,1].

    :param M: Number of hidden units.
    :param scaling: Scaling factor.
    """

    if M <= 0:
        raise ValueError(f"Invalid number of units <<{M}>>. Must be positive.")

    W = scaling * (2 * torch.rand(M, M, dtype=torch.float32, requires_grad=False) - 1)
    W = W - W.T
    return W.to_dense()


def legendre_tensor_init(order: int, theta: float) -> torch.FloatTensor:
    """
    Generate discrete Legendre matrices of order `order` and scaled by `theta`.
    This code derive from the LMU implementation: https://github.com/nengo/keras-lmu/blob/main/keras_lmu/layers.py

    :param order: The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. This parameter sets the number of
        Legendre polynomials used to orthogonally represent the sliding window.
    :param theta: The number of timestamps in the sliding window that is represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented at the time of prediction, however the
        entire sequence will still be processed in order for information to be
        projected to and from the hidden layer. If ``trainable_theta`` is enabled, then
        theta will be updated during the course of training.

    :return: The Matrices A and B of the LTI system.
    """

    Q = torch.arange(order, dtype=torch.float32)
    R = (2 * Q + 1).unsqueeze(1)
    j, i = torch.meshgrid(Q, Q, indexing='ij')
    A = torch.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q.unsqueeze(1) * R

    return continuous_to_discrete_zoh(A.T / theta, B.T / theta)


def continuous_to_discrete_zoh(A, B):
    """
    Function to discretize A and B matrices using Zero Order Hold method.

    Functionally equivalent to
    ``scipy.signal.cont2discrete((A.T, B.T, _, _), method="zoh", dt=1.0)``.
    """

    # combine A/B and pad to make square matrix
    em_upper = torch.cat([A, B], dim=0)
    em = torch.nn.functional.pad(em_upper, (0, B.shape[0]))

    # compute matrix exponential
    ms = torch.matrix_exp(em)

    # slice A/B back out of combined matrix
    discrete_A = ms[: A.shape[0], : A.shape[1]]
    discrete_B = ms[A.shape[0]:, : A.shape[1]]

    return discrete_A, discrete_B


def init_bias(bias: bool, non_linear_units: int, input_scaling: float, bias_scaling: float) -> torch.FloatTensor:
    """
    Initialize the bias tensor.

    :param bias: Whether to use bias or not.
    :param non_linear_units: The number of non-linear units (the dimension of the bias tensor).
    :param input_scaling: The default scaling factor for the bias tensor (if bias_scaling is None).
    :param bias_scaling: The scaling factor for the bias tensor.

    :return: The bias tensor.
    """

    if bias:
        bias_scaling = input_scaling if bias_scaling is None else bias_scaling
        tensor = (2 * torch.rand(non_linear_units, dtype=torch.float32) - 1) * bias_scaling
        tensor = tensor.to_dense()
    else:
        tensor = torch.zeros(non_linear_units, dtype=torch.float32).to_dense()

    return torch.nn.Parameter(tensor, requires_grad=False)


def rescale_kernel(W: torch.FloatTensor, spectral_radius: float, leaky_rate: float, effective_rescaling: bool,
                   distribution: str, non_linear_connectivity: int, non_linear_units: int) -> torch.FloatTensor:
    """
    Rescale the kernel matrix W to have the desired spectral radius.

    :param W: The kernel matrix.
    :param spectral_radius: The desired spectral radius.
    :param leaky_rate: The leaky rate of the kernel.
    :param effective_rescaling: Whether to rescale the kernel considering the leaky rate.
    :param distribution: The distribution from which the kernel was sampled.
    :param non_linear_connectivity: The connectivity of the kernel.
    :param non_linear_units: The number of units in the kernel.

    :return: The rescaled kernel matrix.
    """

    if effective_rescaling and leaky_rate != 1:
        I = eye_init(non_linear_units)
        W = (I * (1 - leaky_rate)) + W * leaky_rate
        W = spectral_norm_scaling(W, spectral_radius)
        return (W + I * (leaky_rate - 1)) * (1 / leaky_rate)
    if distribution == 'normal' and non_linear_units != 1:
        return spectral_radius * W
    if distribution == 'uniform' and non_linear_connectivity == non_linear_units and non_linear_units != 1:
        return fast_spectral_rescaling(W, spectral_radius)

    return spectral_norm_scaling(W, spectral_radius)


def init_non_linear_kernel(non_linear_units: int, non_linear_connectivity: int, distribution: str,
                           spectral_radius: float, leaky_rate: float, effective_rescaling: bool,
                           circular_non_linear_kernel: bool, euler: bool, gamma: float, non_linear_scaling: float) \
        -> torch.FloatTensor:
    """
    Initialize the non-linear kernel.

    :param non_linear_units: The number of units in the non-linear kernel.
    :param non_linear_connectivity: The connectivity of the non-linear kernel.
    :param distribution: The distribution from which the non-linear kernel has to be sampled.
    :param spectral_radius: The desired spectral radius of the non-linear kernel.
    :param leaky_rate: The leaky rate of the non-linear kernel.
    :param effective_rescaling: Whether to rescale the non-linear kernel considering the leaky rate.
    :param circular_non_linear_kernel: Whether to use a non-linear kernel with ring topology.
    :param euler: Whether to use the Euler method to generate the non-linear kernel.
    :param gamma: Scaling factor for the skew-symmetric matrix.
    :param non_linear_scaling: Scaling factor for the non-linear kernel.

    :return: The non-linear kernel matrix.
    """

    if euler:
        W = skewsymmetric(non_linear_units, non_linear_scaling)
        kernel = W - gamma * eye_init(non_linear_units)
    else:
        if circular_non_linear_kernel:
            W = circular_tensor_init(non_linear_units, distribution='fixed', scaling=non_linear_scaling)
            kernel = W
        else:
            W = sparse_tensor_init(non_linear_units, non_linear_units, C=non_linear_connectivity,
                                   distribution=distribution, scaling=non_linear_scaling)
            kernel = rescale_kernel(W, spectral_radius, leaky_rate, effective_rescaling, distribution,
                                    non_linear_connectivity, non_linear_units)
    return torch.nn.Parameter(kernel, requires_grad=False)


def init_input_kernel(input_units: int, units: int, input_connectivity: int, input_scaling: float, distribution: str,
                      signs_from: str | None = None, legendre_input: bool = False, theta: float = 1) \
        -> torch.FloatTensor:
    """
    Initialize the input kernel.

    :param input_units: The number of input dimensions.
    :param units: The number of units in a kernel.
    :param input_connectivity: The connectivity of the input kernel.
    :param input_scaling: The scaling factor for the input kernel.
    :param distribution: The distribution from which the input kernel has to be sampled.
    :param signs_from: The source of the signs for the fixed distribution.
    :param legendre_input: Whether to use the input kernel derived from legendre polynomials.
    :param theta: The scaling factor for the Legendre input kernel.

    :return: The input kernel matrix.
    """

    if legendre_input and input_units == 1:
        _, kernel = legendre_tensor_init(units, theta)
    else:
        if legendre_input and input_units > 1:
            import warnings
            warnings.warn("Legendre input kernel is only available for inputs with 1 dimension."
                          " Default input kernel will be used.")
        kernel = sparse_tensor_init(input_units, units, C=input_connectivity, scaling=input_scaling,
                                    signs_from=signs_from, distribution=distribution)
    return torch.nn.Parameter(kernel, requires_grad=False)


def init_memory_kernel(memory_units: int, theta: float, legendre: bool, scaling: float) -> torch.FloatTensor:
    """
    Initialize the memory kernel.

    :param memory_units: The number of memory units.
    :param theta: The scaling factor for a Legendre matrix.
    :param legendre: Whether to use a Legendre matrix or a circular matrix.
    :param scaling: The scaling factor for the circular matrix.

    :return: The memory kernel matrix.
    """

    if legendre:
        kernel, _ = legendre_tensor_init(memory_units, theta)
    else:
        kernel = circular_tensor_init(memory_units, distribution='fixed', scaling=scaling)
    return torch.nn.Parameter(kernel, requires_grad=False)
