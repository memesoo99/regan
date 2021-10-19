# Utility functions for FewShotCNN
import os
import torch
import yaml
import shutil
import threading
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import pickle

img_save_path = './dataset/images'
label_save_path = './dataset/labels'


def check_dir(folder):
    if os.path.exists(folder) and os.path.isdir(folder):
        return
    try:
        os.mkdir(folder)
    except OSError:
        print("Creation of the directory '%s' failed " % folder)
    else:
        print("Successfully created the main directory '%s' " % folder)
    
    

def save_img_and_label(imgs_gen, predictions,idx):
    imgs_gen = imgs_gen.clamp_(-1., 1.).detach().squeeze().permute(0,2,3,1).cpu().numpy()
    imgs_gen = imgs_gen*0.5 + 0.5
    for i in range(imgs_gen.shape[0]):
        img_filename = f'generated_data_{str(i+idx*10).zfill(6)}.png'
        label_filename = f'generated_label_{str(i+idx*10).zfill(6)}.png'
        plt.imsave(os.path.join(img_save_path,img_filename), imgs_gen[i])
        plt.imsave(os.path.join(label_save_path,label_filename), predictions[i])
        

def merge_config(config,args):
    for key_1 in config.keys():
        if(isinstance(config[key_1],dict)):
            for key_2 in config[key_1].keys():
                if(key_2) in dir(args):
                    config[key_1][key_2] = getattr(args,key_2)
    return config

def load_yaml(yaml_name, args):
    config = yaml.load(open(yaml_name, 'r', encoding='utf-8'),Loader=yaml.FullLoader)
    config = merge_config(config, args)
    return config
    

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def horizontal_concat(imgs):
    return torch.cat([img.unsqueeze(0) for img in imgs], 3) 

def save_generated_image(tensor):
    imgs = tensor.clamp_(-1., 1.).detach().squeeze().permute(0,2,3,1).cpu().numpy()
    imgs = imgs*0.5 + 0.5
    for i in range(imgs.shape[0]):
        filename = f'image_generated_{i}.png'
        plt.imsave(filename, imgs[i])
        

    