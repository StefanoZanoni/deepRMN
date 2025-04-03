import torch

from typing import Callable
from utils.initialization import init_input_kernel, init_non_linear_kernel, init_bias


def validate_params(input_units: int, recurrent_units: int, leaky_rate: float, recurrent_connectivity: int,
                    input_connectivity: int, distribution: str, non_linearity: str, signs_from: str) -> None:
    """
    Validates the parameters for the ReservoirCell.

    :param input_units: Number of input units.
    :param recurrent_units: Number of recurrent units.
    :param leaky_rate: Leaky integration rate.
    :param recurrent_connectivity: Number of connections in the recurrent weight matrix.
    :param input_connectivity: Number of connections in the input weight matrix.
    :param distribution: Distribution type for weight initialization.
    :param non_linearity: Non-linearity function to use.
    :param signs_from: Source for signs of weights.

    :raises ValueError: If any of the parameters are invalid.
    """

    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if recurrent_units < 1:
        raise ValueError("Recurrent units must be greater than 0.")
    if not (0 < leaky_rate <= 1):
        raise ValueError("Leaky rate must be in (0, 1].")
    if not (1 <= recurrent_connectivity <= recurrent_units):
        raise ValueError("Recurrent connectivity must be in [1, recurrent_units].")
    if not (1 <= input_connectivity <= recurrent_units):
        raise ValueError("Input connectivity must be in [1, recurrent_units].")
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")
    if signs_from not in [None, 'random', 'pi', 'e', 'logistic']:
        raise ValueError("Signs from must be None, 'random', 'pi', 'e', or 'logistic'.")


class ReservoirCell(torch.nn.Module):
    """
    A single reservoir cell for the Echo State Network.
    It is a recurrent neural network cell with a leaky integrator or Euler integration.
    The update rule for the reservoir cell with leaky integrator neurons is given by:
     x(t) = (1 - α) * x(t-1) + α * f(W_in * u(t) + W * x(t-1) + b),
    where f is the non-linearity function, W_in is the input weight matrix, W is the recurrent weight matrix, b is the
    bias term, and α is the leaky integration rate.
    The update rule for the reservoir cell with Euler integrator is given by:
     x(t) = x(t-1) + epsilon * f(W_in * u(t) + (W - gamma * I) * x(t-1) + b), where I is the identity matrix, and
    epsilon is the Euler integration step size, and gamma is the diffusion coefficient for the recurrent weights.
    """

    def __init__(self,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 input_scaling: float = 1.0,
                 recurrent_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_recurrent_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 ) -> None:
        """
        Initializes the ReservoirCell.

        :param input_units: Number of input units.
        :param recurrent_units: Number of recurrent units.
        :param input_scaling: Scaling factor for input weights.
        :param recurrent_scaling: Scaling factor for recurrent weights.
        :param spectral_radius: Spectral radius of the recurrent weight matrix.
        :param leaky_rate: Leaky integration rate.
        :param input_connectivity: Number of connections in the input weight matrix.
        :param recurrent_connectivity: Number of connections in the recurrent weight matrix.
        :param bias: Whether to use a bias term.
        :param bias_scaling: Scaling factor for the bias term.
        :param distribution: Distribution type for weight initialization.
        :param signs_from: Source for signs of weights.
        :param fixed_input_kernel: Whether to use a fixed input kernel.
        :param non_linearity: Non-linearity function to use.
        :param effective_rescaling: Whether to rescale the recurrent weights according to the leaky rate.
        :param circular_recurrent_kernel: Whether to use a circular recurrent kernel.
        :param euler: Whether to use Euler integration.
        :param epsilon: Euler integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        """

        super().__init__()

        validate_params(input_units, recurrent_units, leaky_rate, recurrent_connectivity, input_connectivity,
                        distribution, non_linearity, signs_from)

        self.recurrent_units = recurrent_units
        self._leaky_rate = torch.tensor(leaky_rate, dtype=torch.float32, requires_grad=False)
        self._one_minus_leaky_rate = torch.tensor(1 - leaky_rate, dtype=torch.float32, requires_grad=False)

        self.input_kernel = init_input_kernel(input_units, recurrent_units, input_connectivity, input_scaling,
                                              'fixed' if circular_recurrent_kernel and fixed_input_kernel
                                              else distribution,
                                              signs_from=signs_from if circular_recurrent_kernel and fixed_input_kernel
                                              else None)
        self.recurrent_kernel = init_non_linear_kernel(recurrent_units, recurrent_connectivity, distribution,
                                                       spectral_radius, leaky_rate, effective_rescaling,
                                                       circular_recurrent_kernel, euler, gamma, recurrent_scaling)
        self.bias = init_bias(bias, recurrent_units, input_scaling, bias_scaling)

        self._epsilon = torch.tensor(epsilon, dtype=torch.float32, requires_grad=False)
        self._non_linear_function: \
            Callable[[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor | None], torch.FloatTensor] = \
            lambda x, out=None: torch.tanh(x, out=out) if non_linearity == 'tanh' else torch.add(x, 0, out=out)
        self._out = None
        self._state = None
        self._forward_function: Callable[[torch.Tensor], torch.FloatTensor] = (
            self._forward_euler) if euler else self._forward_leaky_integrator

    @torch.no_grad()
    def _forward_leaky_integrator(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass using the leaky integrator method.

        :param xt: Input tensor at time t.

        :return : Updated state tensor.
        """

        # x(t) = (1 - α) * x(t-1) + α * f(W_in * u(t) + W * x(t-1) + b)
        past_state = self._state * self._one_minus_leaky_rate
        self._state = past_state.add_(
            self._non_linear_function(
                torch.addmm(self.bias, self._state, self.recurrent_kernel)
                .addmm_(xt, self.input_kernel), out=self._out
            )
            .mul_(self._leaky_rate)
        )

        return self._state

    @torch.no_grad()
    def _forward_euler(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass using the Euler integration method.

        :param xt: Input tensor at time t.

        :return : Updated state tensor.
        """

        # x(t) = x(t-1) + ε * f(W_in * u(t) + (W - γ * I) * x(t-1) + b)
        self._state += (
            self._non_linear_function(
                torch.addmm(self.bias, self._state, self.recurrent_kernel)
                .addmm_(xt, self.input_kernel), out=self._out
            )
            .mul_(self._epsilon)
        )

        return self._state

    def forward(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass through the reservoir cell.

        :param xt: Input tensor at time t.

        :return : Updated state tensor.
        """

        return self._forward_function(xt)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Resets the state of the reservoir cell.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        self._leaky_rate = self._leaky_rate.to(device)
        self._one_minus_leaky_rate = self._one_minus_leaky_rate.to(device)
        self._epsilon = self._epsilon.to(device)
        self._state = torch.zeros((batch_size, self.recurrent_units), dtype=torch.float32,
                                  requires_grad=False, device=device)
        self._out = torch.empty_like(self._state, requires_grad=False, device=device, dtype=torch.float32)
