import torch
import numpy as np

from torch.utils.data import Dataset


class Narma30(Dataset):
    """
    A PyTorch Dataset class for the NARMA30 dataset.
    """

    def __init__(self, training: bool = False, validation: bool = False, test: bool = False):
        """
        Initializes the NARMA30 dataset.

        :param training: Flag indicating whether the dataset is for training.
        :param validation: Flag indicating whether the dataset is for validation.
        :param test: Flag indicating whether the dataset is for testing.
        """

        data = np.genfromtxt('datasets/narma30/data/narma30.csv', delimiter=',', dtype=np.float32)
        if training and not validation and not test:
            self.data = torch.tensor(data[0][:5000], dtype=torch.float32).unsqueeze(0)
            self.target = torch.tensor(data[1][:5000], dtype=torch.float32).unsqueeze(0)
        elif validation and not training and not test:
            self.data = torch.tensor(data[0][5000:7500], dtype=torch.float32).unsqueeze(0)
            self.target = torch.tensor(data[1][5000:7500], dtype=torch.float32).unsqueeze(0)
        elif test and not training and not validation:
            self.data = torch.tensor(data[0][7500:], dtype=torch.float32).unsqueeze(0)
            self.target = torch.tensor(data[1][7500:], dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError('Invalid dataset request.')

    def __len__(self):
        """
        Returns the length of the dataset.

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
