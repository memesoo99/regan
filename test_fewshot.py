import torch
import argparse
import matplotlib.pyplot as plt
from projector import Projector
from model.stylegan_model import Generator
from model.segmentation_model import FewShotCNN
from utils.auto import load_yaml

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)

device = 'cuda:1'
generator_path = './checkpoint/550000.pt'
FewShotCNN_path = './checkpoint/FewShotCNN.pt'
image_size = 256
latent_dim = 512


projector = Projector(ckpt=generator_path, size = 256,step=100)



#file paths should be given as lists
file = ['./dataset/images/generated_data_000029.png'] #os.listdir() recommended

classes = config['classes']

projected_result = projector.project(file)
sh = list(projected_result.values())[0]['features'].shape[1]

net = torch.load(FewShotCNN_path)

# net = FewShotCNN(sh, len(classes), size='S')
# net.load_state_dict(torch.load(FewShotCNN_path))
device = 'cpu'
net.eval().to(device)

for i in projected_result.keys():

    print(i)
    image_name = i.split('/')[-1].split('.')[0] + '-project_label.png'
    print(image_name)
    features = projected_result[i]['features'].to('cpu')
    print(features.shape)
    torch.cuda.empty_cache()
    out = net(features)
    print("a",features.shape)
    print("b",type(features))
    predictions = out.data.max(1)[1].cpu().numpy()
    print(predictions.shape)
    print(type(predictions))
    print(predictions)
    plt.imsave(image_name, predictions)
    


    
