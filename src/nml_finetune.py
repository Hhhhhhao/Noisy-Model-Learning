import os
import sys 
import numpy as np

import torch

# from src.models.eval import evaluate
# from src.models.finetune import finetune
from src.models.nml_finetune import finetune
from src.models.modeling import ImageClassifier
from src.args import parse_arguments


def noisy_model_learn(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    print(args)
    offsitetuned_checkpoint = finetune(args)


if __name__ == '__main__':
    args = parse_arguments()
    noisy_model_learn(args)
