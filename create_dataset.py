# data creation for auto_shot segmentation
# python create_dataset --config_path 'auto_shot.yaml'
import torch
import argparse
from model.stylegan_model import Generator
from model.segmentation_model import FewShotCNN
from utils.auto import save_img_and_label,check_dir,load_yaml


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)

device = 'cpu'
n_test = 10

image_size = config['image_size']
latent_dim = config['latent_dim']
truncation = config['truncation']
classes = config['classes'].split(',')
class_num = len(classes)

check_dir(config['t_data'])
check_dir(config['l_data'])
    

# features.shape[1] 어떻게 계산하지..???? config에 넣어버려?
net = FewShotCNN(5376, len(classes), size='S')
net = torch.load(config['FewShotCNN'])
net.eval().to(device)


generator = Generator(image_size, latent_dim, 8)
generator_ckpt = torch.load(config['generator_path'], map_location='cpu')
generator.load_state_dict(generator_ckpt["g_ema"], strict=False)
generator.eval().to(device)

#create from 10*x images
x = 3
for i in range(x):
    with torch.no_grad():
        trunc_mean = generator.mean_latent(4096).detach().clone()
        latent = generator.get_latent(torch.randn(n_test, latent_dim, device=device))
        imgs_gen, features = generator([latent],
                                       truncation=truncation,
                                       truncation_latent=trunc_mean.to(device),
                                       input_is_latent=True,
                                       randomize_noise=True)

        torch.cuda.empty_cache()
        out = net(features)
        predictions = out.data.max(1)[1].cpu().numpy()
    
        
        save_img_and_label(imgs_gen, predictions,i)

print(f"Created {10*x} images!")
