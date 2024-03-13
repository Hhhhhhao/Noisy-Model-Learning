import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100

class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 num_shots=None,
                 classnames=None,
                 noise_ratio=None):

        self.train_dataset = PyTorchCIFAR100(
            root=location, download=True, train=True, transform=preprocess
        )
        if num_shots is not None:
            from .common import sample_num_shot
            selected_data, selected_targets = sample_num_shot(self.train_dataset.data, self.train_dataset.targets, 100, num_shots)
            self.train_dataset.data = selected_data
            self.train_dataset.targets = selected_targets
        if noise_ratio is not None:
            from .common import sample_noise_data
            data, noisy_targets = sample_noise_data(self.train_dataset.data, self.train_dataset.targets, 100, noise_ratio)
            self.train_dataset.data = data
            self.train_dataset.targets = noisy_targets

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(
            root=location, download=True, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


