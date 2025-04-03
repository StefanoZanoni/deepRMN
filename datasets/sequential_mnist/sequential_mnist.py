import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


class SequentialMNIST(Dataset):
    """
    A PyTorch Dataset class for the Sequential MNIST dataset.
    """

    def __init__(self, training: bool = True, normalize: bool = False, permute: bool = False, seed: int = None):
        """
        Initializes the SequentialMNIST dataset.

        :param training: Flag indicating whether the dataset is for training or validation/testing.
        :param normalize: Flag indicating whether to normalize the data.
        :param permute: Flag indicating whether to permute the data.
        :param seed: Seed for the random permutation.
        """

        transform_list = [transforms.ToTensor()]
        transform = transforms.Compose(transform_list)

        self.data = datasets.MNIST(root='./datasets/sequential_mnist/data/',
                                   train=training,
                                   download=True,
                                   transform=transform)
        self.data.data = self.data.data.view(-1, 28 * 28).float() / 255.0
        if normalize:
            self.data.data = (self.data.data - 0.1307) / 0.3081
        self.target = self.data.targets.unsqueeze(1).to(torch.int8)
        self.data = self.data.data

        if permute:
            if seed is not None:
                np.random.seed(seed)
            permutation = np.random.permutation(28 * 28)
            self.data = self.data[:, permutation]
        pass

    def __len__(self):
        """
        Returns the length of the dataset.

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
