import torch
import numpy as np

from tqdm import tqdm

from .esn import ReservoirCell

from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from typing import Callable


class DeepEchoStateNetwork(torch.nn.Module):
    """
    Deep Echo State Network model. It is composed of a stack of Echo State Networks (ESNs) and a readout layer.
    """

    def __init__(self,
                 task: str,
                 input_units: int,
                 total_units: int,
                 *,
                 number_of_layers: int = 1,
                 initial_transients: int = 0,
                 input_scaling: float = 1.0,
                 recurrent_scaling: float = 1.0,
                 inter_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 inter_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 concatenate: bool = False,
                 circular_recurrent_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 alphas: list[float] = None,
                 input_to_all: bool = False,
                 ) -> None:
        """
        Initializes the Deep Echo State Network model.

        :param task: The task to perform. It can be either 'classification' or 'regression'.
        :param input_units: Number of input units.
        :param total_units: Number of total units in the reservoir.
        :param number_of_layers: Number of layers in the deep ESN.
        :param initial_transients: Number of initial transients to discard.
        :param input_scaling: Input scaling factor.
        :param recurrent_scaling: Recurrent scaling factor.
        :param inter_scaling: Interlayer scaling factor.
        :param spectral_radius: Spectral radius of the recurrent weight matrix.
        :param leaky_rate: Leaky rate of the neurons.
        :param input_connectivity: Input connectivity.
        :param recurrent_connectivity: Recurrent connectivity.
        :param inter_connectivity: Interlayer connectivity.
        :param bias: Whether to use bias.
        :param bias_scaling: Bias scaling factor.
        :param distribution: Distribution of the weights.
        :param signs_from: Source of the signs of the weights.
        :param fixed_input_kernel: Whether to use a fixed input kernel.
        :param non_linearity: Non-linearity function.
        :param effective_rescaling: Whether to scale the weights considering the leaky rate.
        :param concatenate: Whether to concatenate the reservoir layers states.
        :param circular_recurrent_kernel: Whether to use a circular recurrent kernel.
        :param euler: Whether to use the Euler method.
        :param epsilon: Euler method integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        :param alpha: Regularization strength for the readout layer.
        :param max_iter: Maximum number of iterations for the readout layer.
        :param tolerance: Tolerance for the readout layer.
        """

        super().__init__()
        if alphas is None:
            alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
        if not isinstance(alphas, list):
            raise ValueError('Invalid alphas.')
        self._initial_transients = initial_transients
        self._scaler = None
        self._concatenate = concatenate
        self._initial_transients = initial_transients
        self._total_units = total_units

        # In case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # state variables projected to the next layer fixed;
        # i.e., the number of trainable parameters does not depend on concatenate_non_linear.
        if concatenate:
            self._recurrent_units = max(1, int(total_units / number_of_layers))
            input_connectivity = max(1, int(input_connectivity / number_of_layers))
            recurrent_connectivity = max(1, int(recurrent_connectivity / number_of_layers))
            inter_connectivity = max(1, int(inter_connectivity / number_of_layers))
        else:
            self._recurrent_units = total_units

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            ReservoirCell(
                input_units,
                self._recurrent_units + total_units % number_of_layers if concatenate else self._recurrent_units,
                input_scaling=input_scaling,
                spectral_radius=spectral_radius,
                leaky_rate=leaky_rate,
                input_connectivity=input_connectivity,
                recurrent_connectivity=recurrent_connectivity,
                bias=bias,
                distribution=distribution,
                signs_from=signs_from,
                fixed_input_kernel=fixed_input_kernel,
                non_linearity=non_linearity,
                effective_rescaling=effective_rescaling,
                bias_scaling=bias_scaling,
                circular_recurrent_kernel=circular_recurrent_kernel,
                euler=euler,
                epsilon=epsilon,
                gamma=gamma,
                recurrent_scaling=recurrent_scaling,
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concatenate_non_linear=True
        if concatenate:
            last_h_size = self._recurrent_units + total_units % number_of_layers
        else:
            last_h_size = self._recurrent_units
        for _ in range(number_of_layers - 1):
            reservoir_layers.append(
                ReservoirCell(
                    last_h_size + input_units if input_to_all else last_h_size,
                    self._recurrent_units,
                    input_scaling=inter_scaling,
                    spectral_radius=spectral_radius,
                    leaky_rate=leaky_rate,
                    input_connectivity=inter_connectivity,
                    recurrent_connectivity=recurrent_connectivity,
                    bias=bias,
                    distribution=distribution,
                    signs_from=signs_from,
                    fixed_input_kernel=fixed_input_kernel,
                    non_linearity=non_linearity,
                    effective_rescaling=effective_rescaling,
                    bias_scaling=bias_scaling,
                    circular_recurrent_kernel=circular_recurrent_kernel,
                    euler=euler,
                    epsilon=epsilon,
                    gamma=gamma,
                    recurrent_scaling=recurrent_scaling,
                )
            )
            last_h_size = self._recurrent_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

        if task == 'classification':
            self.readout = RidgeClassifierCV(alphas=alphas)
        elif task == 'regression':
            self.readout = RidgeCV(alphas=alphas)
        self._trained = False
        self._non_linear_states = None
        self._concatenate_state_input = [input_to_all and idx > 0 for idx in range(number_of_layers)]

    def _reset_state(self, batch_size: int, seq_len: int, device: torch.device) -> None:
        """
        Resets the state of the reservoir layers.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        for layer in self.reservoir:
            layer.reset_state(batch_size, device)

        # Pre-allocate layer_states
        self._non_linear_states = [
            torch.empty((batch_size, seq_len, layer.recurrent_kernel.shape[0]),
                        device=device, requires_grad=False, dtype=torch.float32)
            for layer in self.reservoir
        ]

    def _forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the model.

        :param x: The input tensor.

        :return: The state of the deep reservoir for all the time steps and the state for the last time step.
        """

        seq_len = x.shape[1]

        last_non_linear_state = x
        for idx, non_linear_layer in enumerate(self.reservoir):
            layer_states = self._non_linear_states[idx]
            if self._concatenate_state_input[idx]:
                for t in range(seq_len):
                    (layer_states[:, t, :].copy_
                     (non_linear_layer(torch.cat([last_non_linear_state[:, t], x[:, t]], dim=-1))))
            else:
                for t in range(seq_len):
                    layer_states[:, t, :].copy_(non_linear_layer(last_non_linear_state[:, t]))
            last_non_linear_state = layer_states

        if self._concatenate:
            non_linear_states = torch.cat(self._non_linear_states, dim=-1)
        else:
            non_linear_states = self._non_linear_states[-1]

        return non_linear_states[:, self._initial_transients:, :], non_linear_states[:, -1, :]

    def _allocate(self, data: torch.utils.data.DataLoader, use_last_state: bool = True) \
            -> tuple[np.ndarray, np.ndarray, int, int]:
        batch_size = data.batch_size
        num_batches = len(data)
        state_size = self._total_units

        # pre-allocate memory for the states and the targets
        dataset = data.dataset.dataset if isinstance(data.dataset, torch.utils.data.Subset) else data.dataset
        data_attr = getattr(dataset, 'data', None)
        target_attr = getattr(dataset, 'target', None)
        if data_attr is None or target_attr is None:
            raise AttributeError('Dataset does not have the required attributes `data` and `target`.')
        seq_len = data_attr.shape[1]
        states = np.empty((num_batches * batch_size, seq_len - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)
        if len(target_attr.shape) == 2:
            ys = np.empty((num_batches * batch_size, target_attr.shape[1]), dtype=np.float32, order='F')
        else:
            ys = np.empty((num_batches * batch_size, target_attr.shape[1], target_attr.shape[2]),
                          dtype=np.float32, order='F')

        return states, ys, batch_size, seq_len

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        """
        Fits the model to the data.

        :param data: The data to fit the model to.
        :param device: The device to use.
        :param standardize: Whether to standardize the data before training the readout.
        :param use_last_state: Whether to use just the state at the last time step as input to the readout.
        :param disable_progress_bar: Whether to disable the progress bar.
        """

        states, ys, batch_size, seq_len = self._allocate(data, use_last_state)
        self._reset_state(batch_size, seq_len, device)

        if device == torch.device('cuda'):
            torch.cuda.empty_cache()

        self._trained = True
        idx = 0
        try:
            # iterate over the data
            for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
                x = x.to(device)
                states[idx:idx + batch_size] = self._forward(x.unsqueeze(-1) if x.dim() == 2 else x)[1].cpu().numpy() \
                    if use_last_state else self._forward(x.unsqueeze(-1) if x.dim() == 2 else x)[0].cpu().numpy()
                ys[idx:idx + batch_size] = y.numpy()
                idx += batch_size

            if not use_last_state:
                states = np.concatenate(states, axis=0)
                if len(ys.shape) == 3:
                    ys = np.concatenate(ys, axis=0)
                else:
                    ys = np.repeat(ys, states.shape[0] // ys.shape[0], axis=0) if ys.shape[1] == 1 else ys.T

            if standardize:
                self._scaler = StandardScaler().fit(states)
                states = self._scaler.transform(states)

            self.readout.fit(states, ys)
        except Exception as e:
            self._trained = False
            raise e

    @torch.no_grad()
    def score(self, data: torch.utils.data.DataLoader, score_function: Callable[[np.ndarray, np.ndarray], float],
              device: torch.device, standardize: bool = False, use_last_state: bool = True,
              disable_progress_bar: bool = False) -> float:
        """
        Scores the deep echo state network on the given data.

        :param data: The DataLoader for the input data.
        :param score_function: The scoring function.
        :param device: The device to perform computations on.
        :param standardize: Whether to standardize the states before scoring.
        :param use_last_state: Whether to use the state at the last time step for scoring.
        :param disable_progress_bar: Whether to disable the progress bar.

        :return: The score of the deep ESN.
        """

        if not self._trained:
            raise ValueError('The model has not been trained yet. Use the fit method to train the model.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

        states, ys, batch_size, seq_len = self._allocate(data, use_last_state)
        self._reset_state(batch_size, seq_len, device)

        if device == torch.device('cuda'):
            torch.cuda.empty_cache()

        idx = 0
        for x, y in tqdm(data, desc='Scoring', disable=disable_progress_bar):
            x = x.to(device)
            states[idx:idx + batch_size] = self._forward(x.unsqueeze(-1) if x.dim() == 2 else x)[1].cpu().numpy() \
                if use_last_state else self._forward(x.unsqueeze(-1) if x.dim() == 2 else x)[0].cpu().numpy()
            ys[idx:idx + batch_size] = y.numpy()
            idx += batch_size

        if not use_last_state:
            states = np.concatenate(states, axis=0)
            if len(ys.shape) == 3:
                ys = np.concatenate(ys, axis=0)
            else:
                ys = np.repeat(ys, states.shape[0] // ys.shape[0], axis=0) if ys.shape[1] == 1 else ys.T

        if standardize:
            states = self._scaler.transform(states)

        return score_function(self.readout.predict(states), ys)

    @torch.no_grad()
    def predict(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
                use_last_state: bool = True, disable_progress_bar: bool = False) -> np.ndarray:
        """
        Predicts the target values of the data.

        :param data: The data to predict the target values of.
        :param device: The device to use.
        :param standardize: Whether to standardize the data before predicting the target values.
        :param use_last_state: Whether to use just the state at the last time step as input to the readout.
        :param disable_progress_bar: Whether to disable the progress bar.

        :return: The predicted target values of the data.
        """

        if not self._trained:
            raise RuntimeError('Model has not been trained yet.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the scaler has not been fitted yet. Run the fit method'
                                 'with standardize=True first.')

        states, ys, batch_size, seq_len = self._allocate(data, use_last_state)
        self._reset_state(batch_size, seq_len, device)

        if device == torch.device('cuda'):
            torch.cuda.empty_cache()

        idx = 0
        # iterate over the data
        for x, _ in tqdm(data, desc='Predicting', disable=disable_progress_bar):
            x = x.to(device)
            states[idx:idx + batch_size] = self._forward(x.unsqueeze(-1) if x.dim() == 2 else x)[1].cpu().numpy() \
                if use_last_state else self._forward(x.unsqueeze(-1) if x.dim() == 2 else x)[0].cpu().numpy()
            idx += batch_size

        if not use_last_state:
            states = np.concatenate(states, axis=0)

        if standardize:
            states = self._scaler.transform(states)

        return self.readout.predict(states)