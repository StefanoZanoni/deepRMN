import torch
import time

from typing import Callable

from utils.initialization import init_memory_kernel, init_input_kernel, init_non_linear_kernel, init_bias


def validate_params_memory(input_units: int, memory_units: int, input_memory_connectivity: int, distribution: str,
                           signs_from: str, theta: float) -> None:
    """
    Validate the parameters for the memory cell.

    :param input_units: Number of input units.
    :param memory_units: Number of memory units.
    :param input_memory_connectivity: Input to memory connectivity.
    :param distribution: Distribution of the weights.
    :param signs_from: Source of weight signs.
    :param theta: Theta parameter for the Legendre polynomial kernel.
    """

    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if memory_units < 1:
        raise ValueError("Memory units must be greater than 0.")
    if not (1 <= input_memory_connectivity <= memory_units):
        raise ValueError("Input to memory connectivity must be in [1, memory_units].")
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if signs_from not in [None, 'random', 'pi', 'e', 'logistic']:
        raise ValueError("Signs from must be None, 'random', 'pi', 'e', or 'logistic'.")
    if theta <= 0:
        raise ValueError("Theta must be greater than 0.")


def validate_params_non_linear(input_units: int, non_linear_units: int, memory_units: int, leaky_rate: float,
                               memory_non_linear_connectivity: int, input_non_linear_connectivity: int,
                               non_linear_connectivity: int, distribution: str, non_linearity: str,
                               signs_from: str) -> None:
    """
    Validate the parameters for the non-linear cell.

    :param input_units: Number of input units.
    :param memory_units: Number of memory units.
    :param non_linear_units: Number of non-linear units.
    :param leaky_rate: Leaky integrator rate.
    :param memory_non_linear_connectivity: Memory to non-linear connectivity.
    :param input_non_linear_connectivity: Input to non-linear connectivity.
    :param non_linear_connectivity: Non-linear connectivity.
    :param distribution: Distribution of the weights.
    :param non_linearity: Non-linearity function.
    :param signs_from: Source of weight signs.
    """

    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if non_linear_units < 1:
        raise ValueError("Non-linear units must be greater than 0.")
    if memory_units < 1:
        raise ValueError("Memory units must be greater than 0.")
    if not (0 < leaky_rate <= 1):
        raise ValueError("Leaky rate must be in (0, 1].")
    if not (1 <= memory_non_linear_connectivity <= non_linear_units):
        raise ValueError("Memory to non-linear connectivity must be in [1, non_linear_units].")
    if not (1 <= input_non_linear_connectivity <= non_linear_units):
        raise ValueError("Input to non-linear connectivity must be in [1, non_linear_units].")
    if not (1 <= non_linear_connectivity <= non_linear_units):
        raise ValueError("Non-linear connectivity must be in [1, non_linear_units].")
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")
    if signs_from not in [None, 'random', 'pi', 'e', 'logistic']:
        raise ValueError("Signs from must be None, 'random', 'pi', 'e', or 'logistic'.")


class MemoryCell(torch.nn.Module):
    """
    Memory cell for the RMN model.
    """

    def __init__(self,
                 input_units: int,
                 memory_units: int,
                 *,
                 input_memory_scaling: float = 1.0,
                 memory_scaling: float = 1.0,
                 input_memory_connectivity: int = 1,
                 theta: float = 1.0,
                 legendre: bool = False,
                 legendre_input: bool = False,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False, ) \
            -> None:
        """
        Initialize the memory cell.

        :param input_units: Number of input units.
        :param memory_units: Number of memory units.
        :param input_memory_scaling: Input to memory scaling.
        :param memory_scaling: Memory scaling.
        :param input_memory_connectivity: Input to memory connectivity.
        :param theta: Legendre polynomial kernel parameter.
        :param legendre: Whether to use Legendre polynomial kernel.
        :param distribution: Distribution of the weights.
        :param signs_from: Source of weight signs.
        :param fixed_input_kernel: Whether to use fixed input kernel.
        """

        super().__init__()

        validate_params_memory(input_units, memory_units, input_memory_connectivity, distribution, signs_from, theta)

        self.input_memory_kernel = init_input_kernel(
            input_units, memory_units, input_memory_connectivity,
            input_memory_scaling, 'fixed' if fixed_input_kernel else distribution,
            signs_from=signs_from, legendre_input=legendre_input, theta=theta
        )

        self.memory_kernel = init_memory_kernel(memory_units, theta, legendre, memory_scaling)
        self._memory_state = None

    @torch.no_grad()
    def forward(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass for the memory cell.

        :param xt: Input tensor at time t.

        :return: The memory state at time t.
        """

        # m(t) = Vm * m(t-1) + Vx * x(t)
        self._memory_state = torch.addmm(torch.mm(self._memory_state, self.memory_kernel), xt, self.input_memory_kernel)

        return self._memory_state

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Reset the memory state.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        self._memory_state = torch.zeros((batch_size, self.memory_kernel.shape[0]), dtype=torch.float32,
                                         device=device, requires_grad=False)


class NonLinearCell(torch.nn.Module):
    """
    Non-linear cell for the RMN model.
    """

    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 input_non_linear_scaling: float = 1.0,
                 non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_non_linear_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3) -> None:
        """
        Initialize the non-linear cell.

        :param input_units: Number of input units.
        :param non_linear_units: Number of non-linear units.
        :param memory_units: NUmber of memory units.
        :param input_non_linear_scaling: Input to non-linear scaling.
        :param non_linear_scaling: Non-linear scaling.
        :param memory_non_linear_scaling: Memory to non-linear scaling.
        :param input_non_linear_connectivity: Input to non-linear connectivity.
        :param non_linear_connectivity: Non-linear connectivity.
        :param memory_non_linear_connectivity: Memory to non-linear connectivity.
        :param spectral_radius: Desired spectral radius.
        :param leaky_rate: Leaky integrator rate.
        :param bias: Whether to use bias.
        :param bias_scaling: Bias scaling.
        :param distribution: Distribution of the weights.
        :param signs_from: Source of weight signs.
        :param fixed_input_kernel: Whether to use fixed input kernel.
        :param non_linearity: Non-linearity function.
        :param effective_rescaling: Whether to rescale the weights according to the leaky rate.
        :param circular_non_linear_kernel: Whether to use circular non-linear kernel.
        :param euler: Where to use Euler integration.
        :param epsilon: Euler integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        """

        super().__init__()

        validate_params_non_linear(input_units, non_linear_units, memory_units, leaky_rate,
                                   memory_non_linear_connectivity, input_non_linear_connectivity,
                                   non_linear_connectivity, distribution, non_linearity, signs_from)

        self._leaky_rate = torch.tensor(leaky_rate, dtype=torch.float32, requires_grad=False)
        self._one_minus_leaky_rate = torch.tensor(1 - leaky_rate, dtype=torch.float32, requires_grad=False)

        self.input_non_linear_kernel = init_input_kernel(
            input_units, non_linear_units, input_non_linear_connectivity,
            input_non_linear_scaling, 'fixed' if circular_non_linear_kernel and fixed_input_kernel
            else distribution,
            signs_from=signs_from if circular_non_linear_kernel and fixed_input_kernel else None
        )
        self.non_linear_kernel = init_non_linear_kernel(non_linear_units, non_linear_connectivity, distribution,
                                                        spectral_radius, leaky_rate, effective_rescaling,
                                                        circular_non_linear_kernel, euler, gamma,
                                                        non_linear_scaling)
        self._epsilon = torch.tensor(epsilon, dtype=torch.float32, requires_grad=False)
        self.memory_non_linear_kernel = init_input_kernel(memory_units, non_linear_units,
                                                          memory_non_linear_connectivity, memory_non_linear_scaling,
                                                          distribution)
        self.bias = init_bias(bias, non_linear_units, input_non_linear_scaling, bias_scaling)

        self._non_linear_function: \
            Callable[[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor | None], torch.FloatTensor] = \
            lambda x, out=None: torch.tanh(x, out=out) if non_linearity == 'tanh' else torch.add(x, 0, out=out)
        self._out = None

        self._non_linear_state = None
        self._forward_function: Callable[[torch.Tensor, torch.FloatTensor], torch.FloatTensor] = (
            self._forward_euler) if euler else self._forward_leaky_integrator

    @torch.no_grad()
    def _forward_leaky_integrator(self, xt: torch.Tensor,
                                  memory_state: torch.FloatTensor | None = None) -> torch.FloatTensor:
        """
        Forward pass for the non-linear cell using the leaky integrator.

        :param xt: Input tensor at time t.
        :param memory_state: Memory state at time t.
        :return: The non-linear state at time t.
        """

        if memory_state is None:
            # h(t) = (1 - a) * h(t-1) + a * f(Wx * h(t-1) + Wx * x(t) + b)
            past_state = self._non_linear_state * self._one_minus_leaky_rate
            self._non_linear_state = past_state.add_(
                self._non_linear_function(
                    torch.addmm(self.bias, self._non_linear_state, self.non_linear_kernel)
                    .addmm_(xt, self.input_non_linear_kernel), out=self._out
                )
                .mul_(self._leaky_rate)
            )
        else:
            # h(t) = (1 - a) * h(t-1) + a * f(Wx * h(t-1) + Wm * m(t) + Wx * x(t) + b)
            past_state = self._non_linear_state * self._one_minus_leaky_rate
            self._non_linear_state = past_state.add_(
                self._non_linear_function(
                    torch.addmm(self.bias, self._non_linear_state, self.non_linear_kernel)
                    .addmm_(xt, self.input_non_linear_kernel)
                    .addmm_(memory_state, self.memory_non_linear_kernel), out=self._out
                )
                .mul_(self._leaky_rate)
            )

        return self._non_linear_state

    @torch.no_grad()
    def _forward_euler(self, xt: torch.Tensor, memory_state: torch.FloatTensor | None = None) -> torch.FloatTensor:
        """
        Forward pass for the non-linear cell using Euler integration.

        :param xt: Input tensor at time t.
        :param memory_state: Memory state at time t.
        :return: The non-linear state at time t.
        """

        if memory_state is None:
            # h(t) = h(t-1) + ε * f(Wx * x(t) + (W - γ * I) * h(t-1) + b)
            self._non_linear_state += (
                self._non_linear_function(
                    torch.addmm(self.bias, self._non_linear_state, self.non_linear_kernel)
                    .addmm_(xt, self.input_non_linear_kernel), out=self._out
                )
                .mul_(self._epsilon)
            )
        else:
            # h(t) = h(t-1) + ε * f(Wx * x(t) + Wm * m(t) + (W - γ * I) * h(t-1) + b)
            self._non_linear_state += (
                self._non_linear_function(
                    torch.addmm(self.bias, self._non_linear_state, self.non_linear_kernel)
                    .addmm_(xt, self.input_non_linear_kernel)
                    .addmm_(memory_state, self.memory_non_linear_kernel), out=self._out
                )
                .mul_(self._epsilon)
            )

        return self._non_linear_state

    def forward(self, xt: torch.Tensor, memory_state: torch.FloatTensor | None = None) -> torch.FloatTensor:
        """
        Forward pass for the non-linear cell using Euler integration.

        :param xt: Input tensor at time t.
        :param memory_state: Memory state at time t.
        :return: The non-linear state at time t.
        """

        return self._forward_function(xt, memory_state)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Reset the non-linear state.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        self._epsilon = self._epsilon.to(device)
        self._leaky_rate = self._leaky_rate.to(device)
        self._one_minus_leaky_rate = self._one_minus_leaky_rate.to(device)
        self._non_linear_state = torch.zeros((batch_size, self.non_linear_kernel.shape[0]), dtype=torch.float32,
                                             device=device, requires_grad=False)
        self._out = torch.empty_like(self._non_linear_state, device=device, dtype=torch.float32, requires_grad=False)
