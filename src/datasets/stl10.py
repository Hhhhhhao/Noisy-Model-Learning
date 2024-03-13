import os
import torch
import numpy as np
from torchvision.datasets import STL10 as PyTorchSTL10

class STL10:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 num_shots=None,
                 classnames=None,
                 noise_ratio=None):

        self.train_dataset = PyTorchSTL10(
            root=location, download=True, split='train', transform=preprocess
        )
        if num_shots is not None:
            from .common import sample_num_shot
            selected_data, selected_targets = sample_num_shot(self.train_dataset.data, self.train_dataset.labels, 10, num_shots)
            selected_data = np.concatenate(selected_data, axis=0)
            self.train_dataset.data = selected_data
            self.train_dataset.labels = selected_targets

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchSTL10(
            root=location, download=True, split='test', transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


