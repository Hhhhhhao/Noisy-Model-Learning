import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import RenderedSST2 as PyTorchRenderedSST2
from torchvision.datasets import VisionDataset



class RenderedSST2:
    def __init__(self, 
                 preprocess, 
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 num_shots=None,
                 classnames=None,
                 noise_ratio=None):


        self.train_dataset = PyTorchRenderedSST2(
            root=location, download=True, split='train', transform=preprocess
        )
        if num_shots is not None:
            from .common import sample_num_shot
            samples = self.train_dataset._samples
            data, targets = [], []
            for s in samples:
                data.append(s[0])
                targets.append(s[1])
            selected_data, selected_targets = sample_num_shot(data, targets, 2, num_shots)
            new_samples = []
            for d, t in zip(selected_data, selected_targets):
                new_samples.append((d, t))
            self.train_dataset._samples = new_samples

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = PyTorchRenderedSST2(
            root=location, download=True, split='test', transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes