import os
import copy
import time
import tqdm
import random
import numpy as np
import torch

import clip.clip as clip

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.zeroshot import get_zeroshot_classifier
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier, ImageEncoderMLP
from src.models.utils import cosine_lr, torch_load, LabelSmoothing

import src.datasets as datasets


def finetune(args):
    # assert args.load is not None , "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."

    # set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    if args.load is not None:
        image_classifier = ImageClassifier.load(args.load)
    else:
        if args.load_mlp_encoder is not None:
            image_encoder = ImageEncoderMLP.load(args.load_mlp_encoder)
        else:
            image_encoder = ImageEncoder(args, keep_lang=args.model_source in ['clip', 'open_clip'])
        
        if args.model_source == 'timm':
            weights = torch.nn.init.kaiming_uniform(torch.empty((datasets.dataset2classes[args.train_dataset], image_encoder.model.num_features)))
            classification_head = ClassificationHead(False, weights)
        else:
            if args.load_zero_shot_classifier:
                classification_head = get_zeroshot_classifier(args, image_encoder.model)
            else:
                print("re-init classifier head")
                weights = torch.nn.init.kaiming_uniform(torch.empty((datasets.dataset2classes[args.train_dataset], image_encoder.model.embed_dim)))
                classification_head = ClassificationHead(False, weights)
            delattr(image_encoder.model, 'transformer') 
        image_classifier = ImageClassifier(image_encoder, classification_head)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        image_classifier.process_images = False
        print_every = 1000
    elif args.lora:
        import peft
        print('LoRa Fine-tuning')
        # model_named_modules = [(n, type(m)) for n, m in image_classifier.named_modules()]
        # for n in model_named_modules:
        #     print(n)
        
        if args.model_source == 'timm':
            config = peft.LoraConfig(r=8, target_modules=r".*\.mlp\.fc\d", modules_to_save=["classification_head"])
        else:
            config = peft.LoraConfig(r=8, target_modules=r".*\.mlp\.c_fc|.*\.mlp\.c_proj", modules_to_save=["classification_head"])
        
        model = peft.get_peft_model(image_classifier, config)
        model.print_trainable_parameters()
        
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 25
    
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 25
    
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_shots=args.num_shots,
        noise_ratio=args.noise_ratio,
    )
    num_batches = len(dataset.train_loader)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    for epoch in range(args.epochs):
        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

                
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            if args.lora:
                print('Saving model to', args.save)
                image_classifier.save_pretrained(save_directory=args.save)
                model_path = args.save
            else:
                model_path = os.path.join(args.save, f'checkpoint_latest.pt')
                print('Saving model to', model_path)
                image_classifier.save(model_path)
                optim_path = os.path.join(args.save, f'optim_latest.pt')
                torch.save(optimizer.state_dict(), optim_path)

        # Evaluate
        args.current_epoch = epoch
        eval_results = evaluate(image_classifier, args)

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
