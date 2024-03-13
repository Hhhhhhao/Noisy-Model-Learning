# few-shot datasets
from .cifar10 import *
from .cifar100 import *
from .flowers102 import Flowers102
from .food101 import Food101
from .oxford_pet import OxfordIIITPet
from .stanford_car import StanfordCars
from .fgvc_aircraft import FGVCAircraft
from .dtd import DTD
from .country211 import Country211
from .stl10 import STL10
from .sst2 import RenderedSST2


# vtab datasets
from .vtab import Caltech101, EuroSAT, SVHN, PCAM, RESISC45, DiabeticRetinopathy, DMLab


# from .fmow import FMOWID, FMOWOOD, FMOW
from .imagenet import *
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetAValClasses, ImageNetA
from .imagenet_r import ImageNetRValClasses, ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenet_vid_robust import ImageNetVidRobustValClasses, ImageNetVidRobust

# from .iwildcam import IWildCamID, IWildCamOOD, IWildCamIDNonEmpty, IWildCamOODNonEmpty, IWildCam
from .objectnet import ObjectNetValClasses, ObjectNet
# from .ytbb_robust import YTBBRobustValClasses, YTBBRobust
from .domainnet import DomainNetClipart, DomainNetPainting, DomainNetSketch, DomainNetReal


dataset2classes = {
    'CIFAR10': 10,
    'CIFAR101': 10,
    'CIFAR102': 10,
    'CIFAR100': 100,
    'Food101': 101,
    'Flowers102': 102,
    'OxfordIIITPet': 37,
    'StanfordCars': 196,
    'FGVCAircraft': 102,
    'DTD': 47,
    'ImageNet': 1000,
    'ImageNetV2': 1000,
    'ImageNetA': 1000,
    'ImageNetR': 1000,
    'ImageNetSketch': 1000,
    'ImageNetVidRobust':1000,
    'ObjectNet': 1000,
    'STL10': 10,
    'Countray211': 211,
    'Caltech101': 102,
    'EuroSAT': 10,
    'SVHN': 10,
    'Country211': 211,
    'STL10': 10,
    'PCAM': 10,
    'RESISC45': 45,
    'DiabeticRetinopathy': 5,
    'DMLab': 6,
    'RenderedSST2': 2,
    'DomainNetClipart': 345,
    'DomainNetPainting': 345,
    'DomainNetSketch': 345,
    'DomainNetReal': 345,
}


