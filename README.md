# Noisy Model Leanring 

Code for ICLR 2024 Spotlight Paper "[Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks](https://arxiv.org/abs/2309.17002)"


## Pre-trained Models

Our pre-trained models (noisy) are released by by request.
You can also pre-train your own models on noisy datasets using [timm](https://github.com/huggingface/pytorch-image-models) and [open_clip](https://github.com/mlfoundations/open_clip).
Once you have the pre-trained models, replace/create the model config as in:
1. https://github.com/Hhhhhhao/Noisy-Model-Learning/blob/7547d662c9d3c801a10b71072a0dd7beb7e9b481/timm/models/resnet.py#L74
2. https://github.com/Hhhhhhao/Noisy-Model-Learning/blob/7547d662c9d3c801a10b71072a0dd7beb7e9b481/open_clip/open_clip/pretrained.py#L39


## Training 

### Linear Probing

ImageNet-1K Fully-Supervised Models
```
python src/finetune.py --seed=3907 --train-dataset=CIFAR10 --eval-datasets=CIFAR10 --model=resnet50_noise_5 --model_source=timm --epochs=30 --lr=0.01 --wd=0.0001 --batch-size=64 --save SAVE_PATH --results SAVE_PATH/results.jsonl --data-location=DATA_DIR --freeze-encoder  --template=simple_template  --load-zero-shot-classifier False  --num-shots=None
```

CLIP Models
```
python src/finetune.py --seed=3907 --train-dataset=CIFAR10 --eval-datasets=CIFAR10 --model=RN50_yfcc15m_orig --model_source=open_clip --epochs=30 --lr=0.01 --wd=0.0001 --batch-size=64 --save SAVE_PATH --results SAVE_PATH/results.jsonl --data-location=DATA_DIR --freeze-encoder  --template=simple_template  --load-zero-shot-classifier False  --num-shots=None
```


### NMTune 

ImageNet-1K Fully-Supervised Models
```
python src/nml_tune.py --seed=3907 --train-dataset=CIFAR10 --eval-datasets=CIFAR10 --model=resnet50_noise_5 --model_source=timm --epochs=30 --lr=0.001 --wd=0.0001 --batch-size=64 --save SAVE_PATH --results  SAVE_PATH/results.jsonl --data-location=DATA_DIR --template=openai_imagenet_template --freeze-encoder --load-zero-shot-classifier False --use-weak-strong=False --mlp-with-res=True --mlp-res-scale-init=1.0 --mlp-after-ratio=4 --mlp-after-layers=1 --mlp-after-layers=1 --mse-loss-weight=1.0 --cov-loss-weight=0.04 --svd-loss-weight=0.001
```

CLIP Models
```
python src/nml_tune.py --seed=3907 --train-dataset=CIFAR10 --eval-datasets=CIFAR10 --model=RN50_yfcc15m_noise_30 --model_source=open_clip --epochs=30 --lr=0.001 --wd=0.0001 --batch-size=64 --save SAVE_PATH --results  SAVE_PATH/results.jsonl --data-location=DATA_DIR --template=openai_imagenet_template --freeze-encoder --load-zero-shot-classifier False --use-weak-strong=False --mlp-with-res=True --mlp-res-scale-init=1.0 --mlp-after-ratio=4 --mlp-after-layers=1 --mlp-after-layers=1 --mse-loss-weight=1.0 --cov-loss-weight=0.04 --svd-loss-weight=0.001
```


## Bibtex
```
@inproceedings{
chen2024understanding,
title={Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks},
author={Hao Chen and Jindong Wang and Ankit Shah and Ran Tao and Hongxin Wei and Xing Xie and Masashi Sugiyama and Bhiksha Raj},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
}
```


## Acknowledge
Most of our code are developed based on [timm](https://github.com/huggingface/pytorch-image-models), [open_clip](https://github.com/mlfoundations/open_clip), and [wise-ft](https://github.com/mlfoundations/wise-ft)

## Contact
haoc3@andrew.cmu.edu


