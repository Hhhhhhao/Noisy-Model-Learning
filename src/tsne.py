import os
import sys 
import numpy as np

import torch

from src.models.tsne import evaluate_tsne
from src.models.zeroshot import get_zeroshot_classifier
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier, ImageEncoderMLP
from src.args import parse_arguments



def evaluation(args):
    assert args.results_db is not None, 'Please provide a path to store pca results'
    
    if args.load is not None:
        image_classifier = ImageClassifier.load(args.load)
        image_encoder = image_classifier.image_encoder
    else:
        if args.load_mlp_encoder is not None:
            image_encoder = ImageEncoderMLP.load(args.load_mlp_encoder)
            image_encoder.use_residual = args.use_residual
        else:
            image_encoder = ImageEncoder(args, keep_lang=args.model_source in ['clip', 'open_clip'])

    # Load models
    evaluate_tsne(image_encoder, args)

if __name__ == '__main__':
    args = parse_arguments()
    evaluation(args)
