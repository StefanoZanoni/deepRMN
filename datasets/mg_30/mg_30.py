import torch
import numpy as np

from torch.utils.data import Dataset


class MG30(Dataset):
    """
    A PyTorch Dataset class for the MG-30 dataset.
    """

    def __init__(self, training: bool = False, validation: bool = False, test: bool = False):
        """
        Initializes the MG-30 dataset.

        :param training: Flag indicating whether the dataset is for training.
        :param validation: Flag indicating whether the dataset is for validation.
        :param test: Flag indicating whether the dataset is for testing.
        """

        data = np.genfromtxt('datasets/mg_30/data/MG30.csv', delimiter=',', dtype=np.float32)
        training_data = data[:6000]
        validation_data = data[6000:8000]
        test_data = data[8000:]
        if training and not validation and not test:
            self.data = torch.tensor(training_data[:-1], dtype=torch.float32).unsqueeze(0)
            self.target = torch.tensor(training_data[1:], dtype=torch.float32).unsqueeze(0)
        elif validation and not training and not test:
            self.data = torch.tensor(validation_data[:-1], dtype=torch.float32).unsqueeze(0)
            self.target = torch.tensor(validation_data[1:], dtype=torch.float32).unsqueeze(0)
        elif test and not training and not validation:
            self.data = torch.tensor(test_data[:-1], dtype=torch.float32).unsqueeze(0)
            self.target = torch.tensor(test_data[1:], dtype=torch.float32).unsqueeze(0)
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
