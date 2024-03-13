import os
import sys 
import numpy as np
import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.zeroshot import get_zeroshot_classifier
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.args import parse_arguments



def ft(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    finetuned_checkpoint = finetune(args)

    # Load models
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    ft(args)
