import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.integrate import odeint


class Lorenz96(Dataset):
    """
    A PyTorch Dataset class for generating time series data for Lorenz 96 model:
     https://en.wikipedia.org/wiki/Lorenz_96_model.

    Attributes:
        data (torch.Tensor): The input data for the model.
        target (torch.Tensor): The target data for the model.
    """

    def __init__(self, N: int, F: float, dataset_size: int = 1, lag: int = 1):
        """
        Initializes the Lorenz96 dataset.

        Args:
            N: The number of variables in the Lorenz96 model.
            F: The forcing term in the Lorenz96 model.
            dataset_size: The number of examples in the dataset.
            lag: The number of time steps to predict into the future.
        """

        def L96(x: np.ndarray[np.float32], t: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
            """
            The Lorenz 96 model.

            :param x: The initial state of the model
            :param t: Time steps to integrate over

            :return: The state of the model at each time step
            """

            # Setting up vector
            d = np.zeros(N)
            # Loops over indices (with operations and Python underflow indexing handling edge cases)
            for i in range(N):
                d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
            return d

        dt = 0.01
        t = np.arange(0.0, 25 + (lag * dt), dt)
        dataset = []
        for i in range(dataset_size):
            x0 = np.random.rand(N) + F - 0.5  # [F-0.5, F+0.5]
            x0 = x0.astype(np.float32)
            x = odeint(L96, x0, t)
            dataset.append(x)
        dataset = np.stack(dataset, axis=0)
        dataset = torch.from_numpy(dataset).float()

        self.data = dataset[:, :-lag, :]
        self.target = dataset[:, lag:, :]
        pass

    def __len__(self):
        """
        Returns the number of examples in the dataset.

        Returns:
            int: The length of the dataset.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the data and target at the specified index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the data and target tensors.
        """

        return self.data[idx], self.target[idx]
