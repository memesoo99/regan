import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model.stylegan_model import Generator

# python projector.py --ckpt './checkpoint/550000.pt' --size 256 resuzed_me.jpg

class Projector():
    def __init__(self, ckpt, size = 256, lr_rampup = 0.05, lr_rampdown = 0.25, lr = 0.1, noise = 0.05, noise_ramp = 0.75, step = 1000, noise_regularize_ = 1e5, mse = 0,w_plus = False):
        super().__init__()
        
        self.ckpt = ckpt
        self.size = size
        self.lr_rampup = lr_rampup
        self.lr_rampdown = lr_rampdown
        self.lr = lr
        self.noise = noise
        self.noise_ramp = noise_ramp
        self.step = step # over 100
        self.noise_regularize_ = noise_regularize_
        self.mse = mse
        self.w_plus = w_plus
        
    def noise_regularize(self, noises):
        loss = 0

        for noise in noises:
            size = noise.shape[2]

            while True:
                loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
                )

                if size <= 8:
                    break

                noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean([3, 5])
                size //= 2

        return loss


    def noise_normalize_(self, noises):
        for noise in noises:
            mean = noise.mean()
            std = noise.std()

            noise.data.add_(-mean).div_(std)


    def get_lr(self, t, initial_lr, rampdown=0.25, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        return initial_lr * lr_ramp


    def latent_noise(self, latent, strength):
        
        noise = torch.randn_like(latent) * strength
        return latent + noise


    def make_image(self, tensor):
        return (

            tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
        )


    def project(self, files):
        
        device = "cuda"

    
        n_mean_latent = 10000

        resize = min(self.size, 256)

        transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        imgs = []

        for imgfile in files:
            img = transform(Image.open(imgfile).convert("RGB"))
            imgs.append(img)

        imgs = torch.stack(imgs, 0).to(device)

        g_ema = Generator(self.size, 512, 8)
        g_ema_ckpt = torch.load(self.ckpt, map_location='cuda')
        g_ema.load_state_dict(g_ema_ckpt["g_ema"], strict=False)
        g_ema.eval()
        g_ema = g_ema.to(device)

        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if self.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=self.lr)

        pbar = tqdm(range(self.step))
        latent_path = []

        for i in pbar:
            t = i / self.step
            lr = self.get_lr(t, self.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * self.noise * max(0, 1 - t / self.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            n_loss = self.noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = p_loss + self.noise_regularize_ * n_loss + self.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        img_gen, features = g_ema([latent_path[-1]], input_is_latent=True, noise=noises, concat_features=True)

        filename = os.path.splitext(os.path.basename(files[0]))[0] + ".pt"

        img_ar = self.make_image(img_gen)

        result_file = {}
        for i, input_name in enumerate(files):
            noise_single = []
            for noise in noises:
                noise_single.append(noise[i : i + 1])

            result_file[input_name] = {
                "img": img_gen[i],
                "latent": latent_in[i],
                "noise": noise_single,
                "features": features
            }

            img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
            pil_img = Image.fromarray(img_ar[i])
            pil_img.save(img_name)

        torch.save(result_file, filename)
        return result_file
    
