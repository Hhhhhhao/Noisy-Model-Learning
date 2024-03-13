import os
import json
import pickle

import torch
import numpy as np
from tqdm import tqdm 

from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.datasets as datasets


from sklearn.decomposition import TruncatedSVD


def eval_single_dataset(image_encoder, dataset, args):
    args.freeze_encoder = True
    input_key = 'features'
    image_encoder.eval()
    
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_encoder)
    batched_data = enumerate(dataloader)
    device = args.device

    features = []
    targets = []
    with torch.no_grad():
        for i, data in tqdm(batched_data, total=len(dataloader)):
            data = maybe_dictionarize(data)
            x = data[input_key]
            y = data['labels']

            features.append(x.cpu())
            targets.append(y.cpu())

    features = torch.cat(features)
    features = torch.nn.functional.normalize(features, dim=1)
    features = features - features.mean(dim=1, keepdim=True)
    features = features.numpy()
    targets = torch.cat(targets).numpy()

    # run svd
    svd = TruncatedSVD(n_components=features.shape[1])
    svd.fit(features)
    
    components = svd.components_
    # print(u_vectors.shape)
    singular_values = svd.singular_values_
    # print(singular_values.shape)
    # exit(0)
    # pca = PCA()
    # pca.fit(features)
    # singular_values = pca.singular_values_
    # explained_variance_ratio_ = pca.explained_variance_ratio_
    
    return {'components': components, 'singular_values': singular_values, 'labels':targets}


def evaluate_svd(image_classifier, args):
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