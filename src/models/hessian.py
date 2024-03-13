import os
import json
import pickle

import torch
import numpy as np
from tqdm import tqdm 

from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets


from pyhessian import hessian

def eval_single_dataset(image_classifier, dataset, args):
    args.freeze_encoder = True
    if args.freeze_encoder:
        model = image_classifier.classification_head
        input_key = 'features'
        image_enc = image_classifier.image_encoder
    else:
        model = image_classifier
        input_key = 'images'
        image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=True, args=args, image_encoder=image_enc, return_dict=False)
    
    model = model.cuda()
    hes = hessian(model, criterion=torch.nn.CrossEntropyLoss(), dataloader=dataloader, cuda=True)
    eigenvalues, _ = hes.eigenvalues(top_n=50)
    trace = hes.trace()
    return {'hes_eigen': eigenvalues, 'hes_trace': trace}


def evaluate(image_classifier, args):
    if args.eval_datasets is None:
        return
    
    info = vars(args)
    all_results = {}
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )
        results = eval_single_dataset(image_classifier, dataset, args)
        all_results[dataset_name] = results

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db + '.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        # with open(args.results_db, 'a+') as f:
        #     f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    # return info