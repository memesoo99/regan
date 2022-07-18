import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.nn.functional as F
import pickle
import torch
import numpy as np
import time
import random
import copy
import argparse
import cv2
from tqdm import tqdm
from model.segmentation_model import FewShotSeg
from utils.auto import load_yaml



parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
parser.add_argument('--mode', help='person or horse or car. Mode of Model')
args = parser.parse_args()
config = load_yaml(args.config_path, args)

if args.mode == "HUMAN":
    config["MODEL"] = config["MODEL"]["HUMAN"]
elif args.mode == "DOG":
    config["MODEL"] = config["MODEL"]["DOG"]
elif args.mode == "CAT":
    config["MODEL"] = config["MODEL"]["CAT"]
elif args.mode == "WILD":
    config["MODEL"] = config["MODEL"]["WILD"]
elif args.mode == "HORSE":
    config["MODEL"] = config["MODEL"]["HORSE"]


n_samples = config['n_samples']
OUTPUT_PATH = config["MODEL"]['OUTPUT_PATH']
device = config["device"]
model_size = config["model_size"]

with open(config["MODEL"]['data_dir'],"rb") as fw:
    data = pickle.load(fw)
    
classes = config["MODEL"]['classes'].split(',')

net = FewShotSeg(data['features'].shape[1], len(classes), size=model_size)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
labels = torch.tensor(data['labels']).long()
net.train().to(device)
start_time = time.time()

for epoch in tqdm(range(1, 100+1)):
    sample_order = list(range(n_samples))
    random.shuffle(sample_order)

    for idx in sample_order:
        
        sample = data['features'][idx].unsqueeze(0).to(device)
        label = labels[idx].unsqueeze(0).to(device)
        
        out = net(sample)
        
        loss = F.cross_entropy(out, label, reduction='mean')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f'{epoch:5}-th epoch | loss: {loss.item():6.4f} | time: {time.time()-start_time:6.1f}sec')

    scheduler.step()
    torch.save(net, OUTPUT_PATH)  # save trained model
    
print('Done! model saved to ',OUTPUT_PATH)
