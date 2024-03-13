import os
import sys
import numpy as np

import torch

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.zeroshot import get_zeroshot_classifier
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier, ImageEncoderMLP
from src.args import parse_arguments



def evaluation(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    image_encoder = ImageEncoder(args, keep_lang=args.model_source in ['clip', 'open_clip'])
    if args.model_source == 'timm':
        classification_head = ClassificationHead(False, weights=torch.ones((10, 10)))
        image_classifier = ImageClassifier(image_encoder, classification_head)
        pretrained_checkpoint = os.path.join(args.save, 'pretrained.pt')
        image_classifier.save(pretrained_checkpoint)
    else:
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer') 
        image_classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
        image_classifier.save(pretrained_checkpoint)

    # Load models
    if args.load is not None:
        pretrained_checkpoint = args.load
        finetuned = ImageClassifier.load(pretrained_checkpoint)
    elif args.load_mlp_encoder is not None:
        pretrained_checkpoint = args.load_mlp_encoder
        finetuned = ImageClassifier.load(pretrained_checkpoint)
    evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    evaluation(args)
