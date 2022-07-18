# Utility functions for FewShotCNN
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

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

def save_img_and_label(imgs_gen, predictions,idx, img_save_dir, label_save_dir):
    '''
    Save Image and corresponding label

    Parameter 
        imgs_gen : input parameter format, (N, 3, H, W)
        predictions : input parameter format, (N, 3, H, W)
    '''
    imgs_gen = imgs_gen.clamp_(-1., 1.).detach().squeeze().permute(0,2,3,1).cpu().numpy()
    imgs_gen = imgs_gen*0.5 + 0.5
    for i in range(imgs_gen.shape[0]):
        img_filename = f'generated_data_{str(i+idx*10).zfill(6)}.png'
        label_filename = f'generated_label_{str(i+idx*10).zfill(6)}.png'
        plt.imsave(os.path.join(img_save_dir,img_filename), imgs_gen[i])
        plt.imsave(os.path.join(label_save_dir,label_filename), predictions[i].transpose(1,2,0))

def save_generated_image(tensor):
    imgs = tensor.clamp_(-1., 1.).detach().squeeze().permute(0,2,3,1).cpu().numpy()
    imgs = imgs*0.5 + 0.5
    for i in range(imgs.shape[0]):
        filename = f'image_generated_{i}.png'
        plt.imsave(filename, imgs[i])

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

def check_dir(folder):
    if os.path.exists(folder) and os.path.isdir(folder):
        return
    try:
        os.mkdir(folder)
    except OSError:
        print("Creation of the directory '%s' failed " % folder)
    else:
        print("Successfully created the main directory '%s' " % folder)

class Util():
    def __init__(self, n_class):
        self.n_class = n_class
        self.colors = self._sample_colors(self.n_class)
        self.colors[0] = np.array([1., 1., 1.])


    def _sample_colors(self, n=1):
        h = np.linspace(0.0, 1.0, n)[:,np.newaxis]
        s = np.ones((n,1))*0.5
        v = np.ones((n,1))*1.0
        return hsv_to_rgb(np.concatenate([h,s,v], axis=1))
        
    @torch.no_grad()
    def concat_features(features):
        h = max([f.shape[-2] for f in features])
        w = max([f.shape[-1] for f in features])
        return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)    
        
    def get_visualized_label(self, label):
        label_image = np.zeros((label.shape[-2], label.shape[-1],3))

        for c in range(1,self.n_class):
            label_image[label == c] = self.colors[c]

        return label_image.transpose(2,0,1)
    