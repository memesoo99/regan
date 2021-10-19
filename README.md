# Repurposing GANs for segmentation

A simple implementation of https://arxiv.org/abs/2103.04379

  
<br/>



<br/>

### Checklist
Labeller and few-shot model from [Github](https://github.com/bryandlee/repurpose-gan)
- [x] segmentation Labeling Tool
- [x] Projector
- [x] Few-shot Train
- [x] Few-shot Test
- [x] Auto-shot Train
- [] Auto-shot Test

### Result


# How to Use

### Labeller
prepare your dataset by manually labeling the segmentation mask. 
You might need a few, 2~3 train data

### Few-shot Train
FewShotCNN.pt 생성
```
python train_fewshot.py --config_path './auto_shot.yaml'
```

### Few-shot Test
1.projector.py에서 원하는 이미지의 latent vector추출
2.fewshot CNN에 넣음
```
python test_fewshot.py --config auto_shot.yaml
```

### Auto-shot Train
FewShotCNN에서 생성 + labeling한 dataset으로 UNET 훈련
```
python train_autoshot.py --config_path './auto_shot.yaml'
```

### create_dataset
data creation for auto_shot segmentation
5 example data given
```
python create_dataset.py --config_path 'auto_shot.yaml'
```



```
.
├──/checkpoint
|   ├── 550000.pt (pretrained Style-GAN2 generator ckpt)
|   ├── FewShotCNN.pt (pretrained FewShotCNN.pt)
├──/dataset
│   ├── images
│         ├── generated_data_0000001.png
│         └── ...
│   ├── labels
│         ├── generated_label_0000001.png
│         └── ...
│   └── dataset.pkl
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
│   └── auto.py
├──/create_dataset.py
├──/projector.py
├──/auto_shot.yaml
└── ...


```

