import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.projector import Projector
from model.stylegan_model import Generator
from model.segmentation_model import FewShotSeg
from utils.auto import Util, load_yaml, tensor2image, horizontal_concat, imshow, save_img_and_label, check_dir

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

device = config["device"]
generator_path = config["MODEL"]["generator_path"]
FewShotSeg_path = config["MODEL"]["FewShotSeg"]
image_size = config["MODEL"]["image_size"]
latent_dim = config["MODEL"]["latent_dim"]
img_save_dir = config["TEST"]["img_save_dir"]
label_save_dir = config["TEST"]["label_save_dir"]
check_dir(img_save_dir)
check_dir(label_save_dir)
truncation = config["MODEL"]["truncation"]
classes = config["MODEL"]['classes']

util = Util(len(classes))

if  config["TEST"]["own"]:
    projector = Projector(ckpt=generator_path, size = image_size ,step=100)

    # Check FILE
    if os.path.isdir(config["TEST"]["test_data"]):
        file = os.listdir(config["TEST"]["test_data"])
    elif os.path.isfile(config["TEST"]["test_data"]):
        file = [config["TEST"]["test_data"]] 

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
        plt.imsave(os.path.join(img_save_dir,image_name), predictions)

    print(f"projected result saved to {img_save_dir}")

else:
    generator = Generator(image_size, latent_dim, 8)
    generator_ckpt = torch.load(generator_path, map_location='cpu')
    generator.load_state_dict(generator_ckpt["g_ema"], strict=False)
    generator.eval().to(device)

    device = 'cpu'
    n_test = config["TEST"]["n_test"]

    generator.eval().to(device)

    net = torch.load(FewShotSeg_path)
    net.eval().to(device)

    trunc_mean = generator.mean_latent(4096).detach().clone()

    with torch.no_grad():
        latent = generator.get_latent(torch.randn(n_test, latent_dim, device=device))
        imgs_gen, features = generator([latent],
                                    truncation=truncation,
                                    truncation_latent=trunc_mean.to(device),
                                    input_is_latent=True,
                                    randomize_noise=True)

        torch.cuda.empty_cache()
        out = net(features)
        predictions = out.data.max(1)[1].cpu().numpy()

        pred_no_cat = np.zeros_like(imgs_gen)
        for i in range(n_test):
            pred_no_cat[i] = util.get_visualized_label(predictions[i])
            
        if config["TEST"]["save"]:
            save_img_and_label(imgs_gen, pred_no_cat, idx = 0, img_save_dir = img_save_dir, label_save_dir = label_save_dir)
