import os
import sys 
import numpy as np

import torch

from src.models.nml_tune import nml_tune
from src.models.modeling import ImageClassifier
from src.args import parse_arguments


def noisy_model_learn(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    print(args)
    offsitetuned_checkpoint = nml_tune(args)


if __name__ == '__main__':
    args = parse_arguments()
    noisy_model_learn(args)
