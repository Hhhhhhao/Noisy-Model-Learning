import os

import numpy as np

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.models.hessian import evaluate
from src.models.zeroshot import get_zeroshot_classifier
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.args import parse_arguments



def evaluation(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    image_encoder = ImageEncoder(args, keep_lang=args.model_source in ['clip', 'open_clip'])
    if args.model_source == 'timm':
        classification_head = ClassificationHead(False, weights=image_encoder.model.fc.weight, biases=image_encoder.model.fc.bias)
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
    finetuned = ImageClassifier.load(pretrained_checkpoint)
    evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    evaluation(args)
