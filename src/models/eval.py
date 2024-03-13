import os
import json

import torch
import numpy as np
from tqdm import tqdm 
import pickle
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize

import src.datasets as datasets
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report,roc_auc_score

def eval_single_dataset(image_classifier, dataset, args):    
    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc)
    batched_data = enumerate(dataloader)
    device = args.device



    all_labels, all_preds, all_metadata = [], [], []
        
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in tqdm(batched_data, total=len(dataloader)):
            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']
            
            logits = utils.get_logits(x, model)
            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

            all_labels.append(y.cpu().clone().detach())
            all_preds.append(logits.cpu().clone().detach())
            metadata = data['metadata'] if 'metadata' in data else image_paths
            all_metadata.extend(metadata)

        top1 = correct / n
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        all_pred_prob = torch.softmax(all_preds,dim=1)
        all_pred_prob_label = torch.argmax(all_pred_prob,dim=1) 

        if hasattr(dataset, 'post_loop_metrics'):
            #import pdb
            #pdb.set_trace()
            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
            
    if 'top1' not in metrics:
        metrics['top1'] = top1
    
    if args.results_db.endswith('.pkl'):
        if 'confusion' not in metrics:
            metrics['confusion'] = confusion_matrix(all_labels,all_pred_prob_label)
        if 'classification_report' not in metrics:
            metrics['classification_report'] = classification_report(all_labels,all_pred_prob_label)
        if 'precision_recall' not in metrics:
            metrics['precision_recall'] = precision_recall_fscore_support(all_labels,all_pred_prob_label)
        if 'precision_recall_micro' not in metrics:
            metrics['precision_recall_micro'] = precision_recall_fscore_support(all_labels,all_pred_prob_label,average='micro')
        if 'precision_recall_macro' not in metrics:
            metrics['precision_recall_macro'] = precision_recall_fscore_support(all_labels,all_pred_prob_label,average='macro')
        if 'roc_auc_score_micro' not in metrics:
           metrics['roc_auc_score_micro'] = roc_auc_score(all_labels,all_pred_prob_label,average='micro',multi_class='ovr')
        if 'roc_auc_score_macro' not in metrics:
           metrics['roc_auc_score_macro'] = roc_auc_score(all_labels,all_pred_prob_label,average='macro')
    
    return metrics

def evaluate(image_classifier, args, wandb_run=None):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            image_classifier.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )

        results = eval_single_dataset(image_classifier, dataset, args)
        
        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

        if wandb_run is not None:
            log_dict = {'top1': results['top1']}
            wandb_run.log(log_dict)

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        if args.results_db.endswith('.json') or args.results_db.endswith('.jsonl') :
            with open(args.results_db, 'a+') as f:
                f.write(json.dumps(info) + '\n')
        else:
            output = open(args.results_db, 'wb')
            pickle.dump(info, output)
            output.close()
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info
