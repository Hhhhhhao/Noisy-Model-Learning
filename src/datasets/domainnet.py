import os
import PIL
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms


class DomainNetDataset(Dataset):
    def __init__(self, samples, targets, transform=None):
        super().__init__()
        self.samples = samples 
        self.targets = targets 
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.targets[idx]
        
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


class DomainNet:
    def __init__(self, preprocess, 
                 domain='clipart',
                 location=os.path.expanduser('~/data'), 
                 batch_size=128, 
                 num_workers=4, 
                 num_shots=None, 
                 classnames=None,
                 noise_ratio=None):
        
        train_file = os.path.join(location, 'domainnet', f'{domain}_train.txt') 
        test_file = os.path.join(location, 'domainnet', f'{domain}_test.txt') 
        
        train_data, train_targets = [], []
        with open(train_file, 'r') as f:
            for line in f.readlines():
                data, label = line.strip().split(' ')
                data = os.path.join(location, 'domainnet', data)
                train_data.append(data)
                train_targets.append(int(label))
                
        # num shots
        if num_shots is not None:
            from .common import sample_num_shot
            selected_data, selected_targets = sample_num_shot(train_data, train_targets, 345, num_shots)
            self.train_dataset = DomainNetDataset(selected_data, selected_targets, preprocess)
        else:
            self.train_dataset = DomainNetDataset(train_data, train_targets, preprocess)
        
        
        test_data, test_targets = [], []
        with open(test_file, 'r') as f:
            for line in f.readlines():
                data, label = line.strip().split(' ')
                data = os.path.join(location, 'domainnet', data)
                test_data.append(data)
                test_targets.append(int(label))
        self.test_dataset = DomainNetDataset(test_data, test_targets, preprocess)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = np.arange(345)



class DomainNetClipart(DomainNet):
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=4, num_shots=None, classnames=None,
                 noise_ratio=None):
        super().__init__(preprocess, 'clipart', location, batch_size, num_workers, num_shots, classnames,
                 noise_ratio)


class DomainNetSketch(DomainNet):
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=4, num_shots=None, classnames=None,
                 noise_ratio=None):
        super().__init__(preprocess, 'sketch', location, batch_size, num_workers, num_shots, classnames,
                 noise_ratio)
        
        
class DomainNetReal(DomainNet):
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=4, num_shots=None, classnames=None,
                 noise_ratio=None):
        super().__init__(preprocess, 'real', location, batch_size, num_workers, num_shots, classnames,
                 noise_ratio)
        

class DomainNetPainting(DomainNet):
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), batch_size=128, num_workers=4, num_shots=None, classnames=None,
                 noise_ratio=None):
        super().__init__(preprocess, 'painting', location, batch_size, num_workers, num_shots, classnames,
                 noise_ratio)