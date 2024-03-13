import torch
import copy
import math
import torchvision.transforms as transforms

import clip.clip as clip
from timm.models import create_model
from open_clip.open_clip import create_model_and_transforms

from src.models import utils



class TimmWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(model, 'fc'):
            self.num_features = model.fc.in_features
            self.fc = self.model.fc
        elif hasattr(model, 'head'):
            self.num_features = model.head.in_features
            self.fc = self.model.head
        else:
            self.num_features = model.classifier.in_features
            self.fc = self.model.classifier
    
    def encode_image(self, inputs):
        features = self.model.forward_features(inputs)
        features = self.model.forward_head(features, pre_logits=True)
        return features
    

def timm_load(model, dataset):
    model = create_model(model, pretrained=True)
    model_cfg = model.pretrained_cfg
    input_size = model_cfg['input_size'][-1]
    image_size = int(math.ceil(input_size / model_cfg['crop_pct']))
    mean = model_cfg['mean']
    std = model_cfg['std']
    interpolation = model_cfg['interpolation']
    if interpolation == 'bicubic':
        inter_mode = transforms.InterpolationMode.BICUBIC
    elif interpolation == 'bilinear':
        inter_mode = transforms.InterpolationMode.BILINEAR
    else:
        raise NotImplementedError
    
    # warp model as feature extractor
    model = TimmWrapper(model)

    
    scale = (0.8, 1.0)

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, interpolation=inter_mode, scale=scale), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std))])
    transform_val = transforms.Compose([
            transforms.Resize(image_size, interpolation=inter_mode),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std))])
    return model, transform_train, transform_val


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        # TODO: add timm model here
        if args.model_source == 'clip':
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                args.model, args.device, jit=False)
        elif args.model_source == 'timm':
            self.model, self.train_preprocess, self.val_preprocess = timm_load(args.model, args.train_dataset)
        elif args.model_source == 'open_clip':
            model = args.model.split('_')[0]
            pretrained = '_'.join(args.model.split('_')[1:])
            self.model, self.train_preprocess, self.val_preprocess = create_model_and_transforms(
                model,
                pretrained,
                device='cuda'
            )
        else:
            raise NotImplementedError
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)



class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)



class ImageEncoderMLP(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        if args.model_source == 'clip':
            self.model, self.train_preprocess, self.val_preprocess = clip.load(
                args.model, args.device, jit=False)
        elif args.model_source == 'timm':
            self.model, self.train_preprocess, self.val_preprocess = timm_load(args.model, args.train_dataset)
        elif args.model_source == 'open_clip':
            model = args.model.split('_')[0]
            pretrained = '_'.join(args.model.split('_')[1:])
            self.model, self.train_preprocess, self.val_preprocess = create_model_and_transforms(
                model,
                pretrained,
                device='cuda'
            )
        else:
            raise NotImplementedError
        
        self.cache_dir = args.cache_dir
        
        self.mlp_after_ratio = args.mlp_after_ratio
        self.mlp_after_layers = args.mlp_after_layers
        self.return_only_features = False

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
        
        if hasattr(self.model, 'num_features'):
            embed_dim = self.model.num_features
        else:
            embed_dim = self.model.embed_dim

        
        mlp_after = []
        mlp_after_dims = [embed_dim] + [int(embed_dim * self.mlp_after_ratio)] * self.mlp_after_layers + [embed_dim]
        for i in range(len(mlp_after_dims) - 2):
            mlp_after.append(torch.nn.Linear(mlp_after_dims[i], mlp_after_dims[i + 1]))
            mlp_after.append(torch.nn.BatchNorm1d(mlp_after_dims[i + 1]))
            mlp_after.append(torch.nn.ReLU())
        mlp_after.append(torch.nn.Linear(mlp_after_dims[-2], mlp_after_dims[-1]))
        self.mlp_after = torch.nn.Sequential(*mlp_after)
        print(self.mlp_after)

        
        self.mlp_with_res = args.mlp_with_res
        if self.mlp_with_res:
            print("mlp with residuals")
            self.mlp_res_scale_init = args.mlp_res_scale_init
            self.scale_after = torch.nn.Parameter(torch.ones(1, embed_dim) * self.mlp_res_scale_init)
            # self.scale_before = torch.nn.Parameter(torch.ones((1, 3, 1, 1)) * self.mlp_res_scale_init)
    

    def forward(self, images, return_mid_feats=False):
        assert self.model is not None
        
        feats = self.model.encode_image(images)
        mlp_feats = self.mlp_after(feats)
        if self.mlp_with_res:
            mlp_feats = self.scale_after * mlp_feats + feats
        
        if return_mid_feats:
            return feats, mlp_feats
        else:
            return mlp_feats
        

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)
    

class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True, return_mid_feats=False):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess
        self.return_mid_feats = return_mid_feats

    def forward(self, inputs, return_mid_feats=False):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        
        outputs = self.classification_head(inputs)
        
        if return_mid_feats:
            return inputs, outputs
        else:
            return outputs

    def forward_encoder(self, inputs):
        if self.process_images:
            # TODO: make this part better
            if self.return_mid_feats:
                inputs = self.image_encoder(inputs, return_mid_feats=True)
            else:
                inputs = self.image_encoder(inputs)
        return inputs

    def forward_cls_head(self, inputs):
        return self.classification_head(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)