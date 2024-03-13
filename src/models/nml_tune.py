import os
import copy
import time
import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


import clip.clip as clip

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ImageEncoderMLP, ClassificationHead, ImageClassifier, ImageEncoder
from src.models.utils import cosine_lr

import src.datasets as datasets


class TransformTwice:
    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, x):
        x1 = self.transforms1(x)
        x2 = self.transforms2(x)
        return x1, x2
        

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



def nml_tune(args):
    print(args)

    # set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    image_encoder = ImageEncoderMLP(args, keep_lang=args.model_source in ['clip', 'open_clip'])
    # image_encoder = ImageEncoder(args, keep_lang=args.model_source in ['clip', 'open_clip'])
    if args.model_source == 'timm':
        weights = torch.nn.init.kaiming_uniform(torch.empty((datasets.dataset2classes[args.train_dataset], image_encoder.model.num_features)))
    else:
        weights = torch.nn.init.kaiming_uniform(torch.empty((datasets.dataset2classes[args.train_dataset], image_encoder.model.embed_dim)))
        delattr(image_encoder.model, 'transformer') 
    classification_head = ClassificationHead(False, weights)
    image_classifier = ImageClassifier(image_encoder, classification_head)
    image_classifier.return_mid_feats = True
    image_classifier.process_images = True
    
    # freeze image encoder parameters
    if args.freeze_encoder:
        print('Fine-tuning mlp classifier')
        for param in image_encoder.model.parameters():
            param.requires_grad = False
    model = image_classifier
    
    input_key = 'images'
    preprocess_fn = image_encoder.train_preprocess
    strong_preprocess_fn = copy.deepcopy(preprocess_fn)
    if args.use_weak_strong:
        strong_preprocess_fn.transforms.insert(len(preprocess_fn.transforms) - 3, transforms.RandAugment(3, 5, interpolation=transforms.InterpolationMode.BICUBIC))
    # print(strong_preprocess_fn)
    image_enc = None
    print_every = 25
    
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        TransformTwice(preprocess_fn, strong_preprocess_fn),
        location=args.data_location,
        batch_size=args.batch_size,
        num_shots=args.num_shots,
        noise_ratio=args.noise_ratio,
    )
    num_batches = len(dataset.train_loader)

    model = model.cuda()
    # image_classifier.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()
    
    print("start training")
    mse_loss_fn = torch.nn.MSELoss()
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    
    mse_loss_weight = args.mse_loss_weight
    cov_loss_weight = args.cov_loss_weight
    svd_loss_weight = args.svd_loss_weight
    
    # params = [p for p in model.parameters() if p.requires_grad] + [p for p in image_classifier.parameters()]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    for epoch in range(args.epochs):
        # model.cuda()
        model.train()
        model.module.image_encoder.model.eval()
        # image_classifier.cuda()
        # image_classifier.train()
        
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()
            
            if isinstance(batch, dict):
                inputs_tuple = batch['images']
                labels = batch['labels']
            else:
                inputs_tuple, labels = batch
            inputs_w, inputs_s = inputs_tuple
            inputs_w = inputs_w.cuda()
            inputs_s = inputs_s.cuda()
            labels = labels.cuda()
            inputs = torch.cat([inputs_w, inputs_s], dim=0)
            data_time = time.time() - start_time
            
            # features
            feats, mlp_feats = model.module.forward_encoder(inputs)
            feats_w, _ = feats.chunk(2)
            mlp_feats_w, mlp_feats_s = mlp_feats.chunk(2)
            
            # ce loss
            logits = model.module.forward_cls_head(mlp_feats)
            labels = torch.cat([labels, labels], dim=0)
            ce_loss = ce_loss_fn(logits, labels)
            
            # total loss
            loss = ce_loss
            
            if mse_loss_weight:
                mse_loss = mse_loss_fn(F.normalize(mlp_feats_s, dim=1), F.normalize(feats_w.detach(), dim=1))
                loss += mse_loss_weight * mse_loss 
            else:
                # print("no mse")
                mse_loss = torch.Tensor([0.0])

            if svd_loss_weight:
                u, s, v = torch.svd(mlp_feats)
                s = s / torch.sum(s)
                loss -= svd_loss_weight * torch.mean(s[:5])

            if cov_loss_weight:
                mlp_feats = mlp_feats - mlp_feats.mean(dim=0)
                cov_mlp_feats = (mlp_feats.T @ mlp_feats) / (mlp_feats.size(0) - 1)
                cov_loss = off_diagonal(cov_mlp_feats).pow_(2).sum().div(mlp_feats.size(1))
                loss += cov_loss_weight * cov_loss
            else:
                # print("no conv")
                cov_loss = torch.Tensor([0.0])

            # if var_loss_weight:
            #     std_feats = torch.sqrt(mlp_feats.var(dim=0) + 0.0001)
            #     std_loss = torch.mean(F.relu(1 - std_feats)) / 2
            #     loss += var_loss_weight * std_loss
            # else:
            #     print("no std")
            #     std_loss = torch.Tensor([0.0])
                
            loss.backward()
            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
                # if wandb_run is not None:
                #     wandb_run.log({
                #         'train/loss': loss.item(),
                #         'train/mse_loss': mse_loss.item(),
                #         'train/ce_loss': ce_loss.item(),
                #         'train/cov_loss': cov_loss.item(),
                #         'train/std_loss': std_loss.item(),
                #     }, step=epoch*num_batches + i)
                    

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            # model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            model_path = os.path.join(args.save, f'checkpoint_latest.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            # optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            optim_path = os.path.join(args.save, f'optim_latest.pt')
            torch.save(optimizer.state_dict(), optim_path)

        # Evaluate
        args.current_epoch = epoch
        # image_encoder.eval()
        # image_classifier.eval()
        
        # if args.freeze_encoder:
        #     image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        # else:
        image_classifier = model.module
        # image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        # eval_results = evaluate(ImageClassifier(image_encoder, image_classifier), args)
        eval_results = evaluate(image_classifier, args)
        
        
    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    offsitetune(args)
