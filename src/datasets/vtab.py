

import os
import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from clip_benchmark.datasets.builder import build_vtab_dataset, _load_classnames_and_classification_templates



class FewShotDataset(Dataset):
    def __init__(self, dataset, num_classes, preprocess, num_shots=None):
        data = []
        targets = []
        for image, label in dataset:
            data.append(image)
            targets.append(label)
        
        if num_shots is not None:
            from .common import sample_num_shot
            selected_data, selected_targets = sample_num_shot(data, targets, num_classes, num_shots)
            self.data = selected_data
            self.targets = selected_targets
        else:
            self.data = data
            self.targets = targets
        
        self.preprocess = preprocess

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        data = self.preprocess(data)
        return data, target

    def __len__(self):
        return len(self.data)
    

class Caltech101:
    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('caltech101', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('caltech101', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes

    

class DMLab:
    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('dmlab', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('dmlab', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes


class DiabeticRetinopathy:
    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('diabetic_retinopathy', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('diabetic_retinopathy', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes


class EuroSAT:
    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('eurosat', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        print(len(train_dataset.classes))
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('eurosat', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes


class SVHN:
    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('svhn', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        print(len(train_dataset.classes))
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('svhn', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes


class RESISC45:
    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('resisc45', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        print(len(train_dataset.classes))
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('resisc45', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes

class PCAM:
    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('pcam', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        print(len(train_dataset.classes))
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('pcam', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes


    def __init__(self, 
                preprocess,
                location=os.path.expanduser('~/data'),
                batch_size=128,
                num_workers=4,
                num_shots=None,
                classnames=None,
                noise_ratio=None):

        location = os.path.join(location, 'vtab')
        current_folder = os.path.dirname(__file__)
        classnames, templates = _load_classnames_and_classification_templates('cifar10', current_folder, "en")
        train_dataset = build_vtab_dataset('svhn', download=True, split='trainval', data_dir=location, transform=None, classnames=classnames)
        self.train_dataset = FewShotDataset(train_dataset, num_classes=len(train_dataset.classes), preprocess=preprocess, num_shots=num_shots)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        test_dataset = build_vtab_dataset('svhn', download=True, split='test', data_dir=location, transform=None, classnames=classnames)
        self.test_dataset = FewShotDataset(test_dataset, num_classes=len(test_dataset.classes), preprocess=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.classnames = test_dataset.classes

