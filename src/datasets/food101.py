import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import Food101 as PyTorchFood101
from torchvision.datasets import VisionDataset

from sklearn.metrics import balanced_accuracy_score

class Food101:
    def __init__(self, 
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 num_shots=None,
                 classnames=None,
                 noise_ratio=None):


        self.train_dataset = PyTorchFood101(
            root=location, download=True, split='train', transform=preprocess
        )
        if num_shots is not None:
            from .common import sample_num_shot
            selected_data, selected_targets = sample_num_shot(self.train_dataset._image_files, self.train_dataset._labels, 101, num_shots)
            self.train_dataset._image_files = selected_data
            self.train_dataset._labels = selected_targets

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchFood101(
            root=location, download=True, split='test', transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = np.arange(101)