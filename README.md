
# Repurposing GANs for segmentation

A simple implementation of https://arxiv.org/abs/2103.04379

For styleGAN generator ckpt `checkpoint/550000.pt` checkout -> https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view (FFHQ)
  
<br/>



<br/>

### Checklist
Labeller from @bryandlee [Github](https://github.com/bryandlee/repurpose-gan)
- [x] segmentation Labeling Tool
- [x] Projector
- [x] Few-shot Train
- [x] Few-shot Test
- [x] Auto-shot Train
- [ ] Auto-shot Test

### Result

![generated_data_000003](https://user-images.githubusercontent.com/68745418/137844467-47e27a6b-b03d-449d-8072-deee9756b203.png)
![generated_label_000003](https://user-images.githubusercontent.com/68745418/137844477-a70f4a6c-7c23-4ed7-8dd7-f46b658f70fe.png)
![generated_data_000002](https://user-images.githubusercontent.com/68745418/137866319-cba203e4-f8c2-4a0c-b4d9-621e4099c36e.png)
![generated_label_000002](https://user-images.githubusercontent.com/68745418/137866208-fb8e76b8-a9d5-478a-a734-eb0a160818d6.png)


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
python tools/test_fewshot.py --config './auto_shot.yaml' --mode 'HORSE'
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
https://colab.research.google.com/github/dvschultz/stylegan2-ada-pytorch/blob/main/SG2_ADA_PT_to_Rosinality.ipynb