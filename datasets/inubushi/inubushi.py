import torch
from torch.utils.data import Dataset


class Inubushi(Dataset):
    """
    A PyTorch Dataset class for generating time series data for Inubushi-Yoshimura experiments.

    Attributes:
        data (torch.Tensor): The input data for the model.
        target (torch.Tensor): The target data for the model.
    """

    def __init__(self, k: int, v: float, training: bool = True):
        """
        Initializes the Inubushi dataset.

        Args:
            k (int): The delay parameter for the time series.
            v (float): The non-linearity parameter.
            training (bool): Flag indicating whether the dataset is for training or validation/testing.
        """

        if training:
            sequence_length = 4000 + k
        else:
            sequence_length = 1000 + k

        time_series = 2 * torch.rand(sequence_length) - 1
        self.data = time_series[k:].unsqueeze(0)
        self.target = torch.sin(v * time_series[:-k]).unsqueeze(0)

    def __len__(self):
        """
        Returns the number of examples in the dataset.

        Returns:
            int: The length of the dataset.
        """

        return 1

    def __getitem__(self, idx):
        """
        Retrieves the data and target at the specified index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the data and target tensors.
        """

        return self.data[idx], self.target[idx]
