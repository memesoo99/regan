
# Repurposing GANs for segmentation

A simple implementation of https://arxiv.org/abs/2103.04379

For all the `pretrained GAN generator` and `1-shot dataset sample`, visit [Drive](https://drive.google.com/drive/folders/1uwazXGD_gxADuFRCRhB2McnT1kUFYf4v?usp=sharing)
  
<br/>
<br/>

## Checklist
Labeller from @bryandlee
- [x] segmentation Labeling Tool
- [x] Projector
- [x] Few-shot Train
- [x] Few-shot Test
- [x] Auto-shot Train
- [ ] Auto-shot Test

## Result
- All of the results are based on 1-shot segmentation. For more delicate results, enlarge 1-shot to n-shot (maybe 2-3)
![스크린샷 2022-07-12 오후 2 39 54](https://user-images.githubusercontent.com/68745418/178416890-085e6c45-9882-4ca6-a60c-3b4a2f6ad71b.png)

![스크린샷 2022-07-12 오후 2 41 11](https://user-images.githubusercontent.com/68745418/178417003-86a9714c-78d3-4a50-b373-a75703bf641f.png)

# Requirements
- Pytorch 1.12.0
- CUDA 11.6
- supports single GPU

# How to Use

## 1. Prepare Your Dataset
### Labeller
- prepare your dataset by manually labeling the segmentation mask. 
- You might need a few, 1~3 train data.
- `Provided labeled dataset are all 1-shot`

## 2. Train your few-shot segmentation Model
### Few-shot Train
```
python tools/train_fewshot.py --config_path './auto_shot.yaml' --mode 'HUMAN'
```

### Few-shot Test
- Segment your own custom image(not GAN generated image).
```
python tools/test_fewshot.py --config './auto_shot.yaml'
```

### create_dataset
data creation for auto_shot segmentation
```
python utils/create_dataset.py --config_path './auto_shot.yaml' --mode 'HUMAN'
```

### Auto-shot Train
Train UNET with created dataset
```
python tools/train_autoshot.py --config_path './auto_shot.yaml'
```

```
.
├──/checkpoint
|   └── pretrained StyleGAN2 weights 
├──/dataset
│   ├── CAT
│         └── cat_1shot.pkl
│   ├── DOG
│   ├── HUMAN
│   └── WILD
├──/model
│   ├── segmentation_model.py
│   ├── stylegan_model.py
│   └── Unet.py
├──/loss
│   └── losses.py
├──/metric
│   └── Metrics.py
├──/utils
│   ├── 2d_from_3d.py
│   ├── projector.py
│   ├── create_dataset.py.py
│   └── auto.py
├──/tools
│   ├── Data_Loader.py
│   ├── train_fewshot.py
│   ├── train_autoshot.py
│   └── test_fewshot.py
├──/auto_shot.yaml
└── ...


```

### Useful Site
[SG2-ADA-PT to Rosinality.ipynb](https://colab.research.google.com/github/dvschultz/stylegan2-ada-pytorch/blob/main/SG2_ADA_PT_to_Rosinality.ipynb)
