import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import argparse
import matplotlib.pyplot as plt
from utils.projector import Projector
from model.stylegan_model import Generator
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

device = config["device"]
generator_path = config["MODEL"]["generator_path"]
FewShotSeg_path = config["MODEL"]["FewShotSeg"]
image_size = config["MODEL"]["image_size"]
latent_dim = config["MODEL"]["latent_dim"]
save_dir = config["TEST"]["save_dir"]

projector = Projector(ckpt=generator_path, size = image_size ,step=100)

# Check FILE
if os.path.isdir(config["TEST"]["test_data"]):
    file = os.listdir(config["TEST"]["test_data"])
elif os.path.isfile(config["TEST"]["test_data"]):
    file = [config["TEST"]["test_data"]] 

classes = config["MODEL"]['classes']

projected_result = projector.project(file)
sh = list(projected_result.values())[0]['features'].shape[1]

net = torch.load(FewShotSeg_path)

device = 'cpu'
net.eval().to(device)

for i in projected_result.keys():
    image_name = i.split('/')[-1].split('.')[0] + '-project_label.png'
    features = projected_result[i]['features'].to('cpu')
    torch.cuda.empty_cache()
    out = net(features)
    predictions = out.data.max(1)[1].cpu().numpy()
    plt.imsave(os.path.join(save_dir,image_name), predictions)

print(f"projected result saved to {save_dir}")
    


    
