import os
import torch
from torchvision.datasets import Country211 as PyTorchCountry211

class Country211:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 num_shots=None,
                 classnames=None,
                 noise_ratio=None):

        self.train_dataset = PyTorchCountry211(
            root=location, download=True, split='train', transform=preprocess
        )
        if num_shots is not None:
            from .common import sample_num_shot
            
            all_data = []
            all_targets = []
            for data, target in self.train_dataset.samples:
                all_data.append(data)
                all_targets.append(target)
            selected_data, selected_targets = sample_num_shot(all_data, all_targets, 211, num_shots)
            new_samples = [(d, t) for d, t in zip(selected_data, selected_targets)]
            self.train_dataset.samples = new_samples
            self.train_dataset.imgs = new_samples
            self.train_dataset.targets = selected_targets

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCountry211(
            root=location, download=True, split='test', transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = self.test_dataset.classes


