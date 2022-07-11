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
from utils.auto import load_yaml
from model.segmentation_model import FewShotCNN


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)

n_samples = 2
PATH = config['PATH']
device = 'cuda'


with open(config['data_dir'],"rb") as fw:
    data = pickle.load(fw)
    

classes = config['classes'].split(',')

net = FewShotCNN(data['features'].shape[1], len(classes), size='S')

optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.Step LR(optimizer, step_size=50, gamma=0.1)
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
    torch.save(net, PATH)  # 전체 모델 저장
print('Done! model saved to ',PATH)
