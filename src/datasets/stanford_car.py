import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import StanfordCars as PyTorchStanfordCars

from sklearn.metrics import balanced_accuracy_score

class StanfordCars:
    def __init__(self, 
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 num_shots=None,
                 classnames=None,
                 noise_ratio=None):


        self.train_dataset = PyTorchStanfordCars(
            root=location, download=True, split='train', transform=preprocess
        )
        if num_shots is not None:
            from .common import sample_num_shot
            data, targets = [], []
            for d, t in self.train_dataset._samples:
                data.append(d)
                targets.append(t)
            selected_data, selected_targets = sample_num_shot(data, targets, 196, num_shots)
            samples = []
            for d, t in zip(selected_data, selected_targets):
                samples.append((d, t))
            self.train_dataset._samples = samples

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchStanfordCars(
            root=location, download=True, split='test', transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = np.arange(196)