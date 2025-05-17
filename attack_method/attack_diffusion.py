from typing import Union, List, Optional, Callable
import comet_ml
from comet_ml import Experiment
import torch
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from diffusers import StableDiffusionInstructPix2PixPipeline
import requests
from io import BytesIO
from huggingface_hub import login
import piq
import os
import statistics
import math
import argparse

# Configure comet ml experiments here
PROJECT_NAME = None
# comet_ml.login(project_name=PROJECT_NAME ,api_key="Enter your api key")

# configure huggingface_hub login token
# TOKEN = # Enter your private token here
TOKEN = None


def preprocess(image):
    """
    Input: PIL Image (H, W, C)
    Output: Torch Tensor (1, C, H, W)
    Description: Resize image to a multiple of 32.
    Convert from [0, 255] to [0, 1], then [-1, 1] for VAE
    """

    w, h = image.size
    # Resize to integer multiples of 32
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def prepare_image(image):
    """
    Input: PIL Image
    Output: PyTorch Tensor (1, C, H, W)
    Description: convert PIL Image to PyTorch Tensor
    Convert from [0, 255] to [-1, 1]
    """

    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


def attack_forward(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
    ):

    text_inputs = self.tokenizer(prompt, 
                                 padding="max_length",
                                 max_length=self.tokenizer.model_max_length,
                                 return_tensors="pt")
    text_input_ids = text_inputs.input_ids
    text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

    uncond_tokens = [""]
    max_length = text_input_ids.shape[-1]
    uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    seq_len = uncond_embeddings.shape[1]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    text_embeddings = text_embeddings.detach()

    num_channels_latents = self.vae.config.latent_channels
    latents_shape = (1 , num_channels_latents, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)

    image_latents = self.vae.encode(image).latent_dist.sample()
    image_latents = 0.18215 * image_latents
    image_latents = torch.cat([image_latents] * 2)

    latents = latents * self.scheduler.init_noise_sigma

    self.scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)

    for i, t in enumerate(timesteps_tensor):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    image = self.vae.decode(latents).sample

    return image



def compute_grad(image, prompt, target_image, pipe, **kwargs):
    
    torch.set_grad_enabled(True)
    cur_img = image.clone()
    cur_img.requires_grad_()
    image_nat = attack_forward(pipe, image=cur_img, prompt=prompt, **kwargs)
    loss = (image_nat - target_image).norm(p=2)
    grad = torch.autograd.grad(loss, [cur_img])[0]

    return grad, loss.item(), image_nat.data.cpu()




def diffusion_l2(X, prompt, step_size, iters, eps, 
             clamp_min, clamp_max, pipe, grad_reps = 5, target_image = 0, experiment=None, **kwargs):
    
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    for index in iterator:
        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(X_adv, prompt, target_image=target_image, pipe=pipe, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)
        if experiment:
            experiment.log_metric("loss", sum(losses)/len(losses), step=index)
        
        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        actual_step_size = step_size - (step_size - step_size / 100) / iters * index
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)

    torch.cuda.empty_cache()
    
    return X_adv, last_image



def diffusion_linf(X, prompt, step_size, iters, eps, 
               clamp_min, clamp_max, pipe, grad_reps = 5, target_image = 0, experiment=None, **kwargs):
    
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    for index in iterator:
        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(X_adv, prompt, target_image=target_image, pipe=pipe, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)
        if experiment:
            experiment.log_metric("loss", sum(losses)/len(losses), step=index)

        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        with torch.no_grad():

            actual_step_size = step_size - (step_size - step_size / 100) / iters * index
            X_adv = X_adv - grad.detach().sign() * actual_step_size

            X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
            X_adv.grad = None
    
    torch.cuda.empty_cache()

    return X_adv, last_image






def encoder_pgd_l2(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, experiment=None):
    """
    Input:
        X: input image as Torch Tensor
        model: model being attacked
        eps: attack budget
        step_size: gradient descent step size
        iters: number of gradient descent iterations
        clamp_min, clamp_max: minimum and maximum value
    Output:
        X_adv: input image with adversarial perturbation
    Description:
        Encoder attack implemented using PGD with l_2 norm.
    """

    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        #actual_step_size = step_size
        X_adv.requires_grad_(True)
        loss = (model(X_adv).latent_dist.mean).norm()
        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
        grad, = torch.autograd.grad(loss, [X_adv])
        with torch.no_grad():
            X_adv = X_adv - grad.detach() / torch.norm(grad.detach()) * actual_step_size
            d_x = X_adv - X.detach()
            d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
            X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)
            X_adv.grad = None

        if experiment:
            experiment.log_metric("loss", loss, step=i)

    return X_adv



def encoder_pgd_inf(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, experiment=None):
    """
    Input:
        X: input image as Torch Tensor
        model: model being attacked
        eps: attack budget
        step_size: gradient descent step size
        iters: number of gradient descent iterations
        clamp_min, clamp_max: minimum and maximum value
    Output:
        X_adv: input image with adversarial perturbation
    Description:
        Encoder attack implemented using PGD with l_inf norm.
    """

    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        #actual_step_size = step_size
        X_adv.requires_grad_(True)
        loss = (model(X_adv).latent_dist.mean).norm()
        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
        grad, = torch.autograd.grad(loss, [X_adv])
        with torch.no_grad():
            X_adv = X_adv - grad.detach().sign() * actual_step_size
            X_adv = torch.minimum(torch.maximum(X_adv, X-eps), X+eps)
            X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
            X_adv.grad = None

        if experiment:
            experiment.log_metric("loss", loss, step=i)

    return X_adv




def encoder_pgd_l1(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, experiment=None):

    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    d_x = X_adv - X.detach()
    d_x_norm = torch.renorm(d_x, p=1, dim=0, maxnorm=eps)
    X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)

    
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        #actual_step_size = step_size
        X_adv.requires_grad_(True)
        loss = (model(X_adv).latent_dist.mean).norm()
        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
        grad, = torch.autograd.grad(loss, [X_adv])
        with torch.no_grad():
            X_adv = X_adv - grad.detach() / (torch.norm(grad.detach(), p=1) + 1e-10) * actual_step_size
            d_x = X_adv - X.detach()
            
            d_x_norm = torch.renorm(d_x, p=1, dim=0, maxnorm=eps)
            X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)
            X_adv.grad = None

        if experiment:
            experiment.log_metric("loss", loss, step=i)

    return X_adv


def encoder_fgsm(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, experiment=None):
    """
    Input:
        X: input image as Torch Tensor
        model: model being attacked
        eps: attack budget
        step_size: gradient descent step size
        iters: number of gradient descent iterations
        clamp_min, clamp_max: minimum and maximum value
    Output:
        X_adv: input image with adversarial perturbation
    Description:
        Encoder attack implemented using PGD with l_inf norm.
    """

    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    actual_step_size = eps
    X_adv.requires_grad_(True)
    loss = (model(X_adv).latent_dist.mean).norm()
    grad, = torch.autograd.grad(loss, [X_adv])
    with torch.no_grad():
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X-eps), X+eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None

    if experiment:
        experiment.log_metric("loss", loss)

    return X_adv



def display_images(init_image, adv_image, image_nat, image_adv, prompt, SEED):
    """
    Input:
        init_image: initial clean image
        adv_image: initial image with adversarial perturbation
        image_nat: modified original image
        image_adv: modified adversarial image
        prompt: image editing prompt
        SEED: image editing seed
    Output:
        None
    Description:
        Plot the four images in a row.
    """

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,6))
    
    ax[0].imshow(init_image)
    ax[1].imshow(adv_image)
    ax[2].imshow(image_nat)
    ax[3].imshow(image_adv)

    ax[0].set_title('Source Image', fontsize=16)
    ax[1].set_title('Adversarial Image', fontsize=16)
    ax[2].set_title('Generated Image Natural.', fontsize=16)
    ax[3].set_title('Generated Image Adversarial.', fontsize=16)

    for i in range(4):
        ax[i].grid(False)
        ax[i].axis('off')
    
    fig.suptitle(f"Prompt: {prompt} | Seed:{SEED}", fontsize=20)
    fig.tight_layout()
    plt.show()



def load_images(folder=None):
    if not folder:
        image_dir = os.path.join(os.path.dirname(__file__), 'images')
    else:
        image_dir = folder
    image_list = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert('RGB')
            image_list.append(img)
    return image_list



def encoder_attack(method, project_name=None, path=None):
    """Encoder attack on a single image."""
    
    experiment = None
    
    login(token=TOKEN)
    to_pil = T.ToPILImage()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    if path:
        init_image = Image.open(path).convert('RGB')
    else:
        init_image = Image.open('003004.jpg').convert('RGB')
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    init_image = center_crop(resize(init_image))

    # Set the prompt and seed
    prompt = "Let the woman wear glasses."
    SEED = 9999

    with torch.autocast('cuda'):
        X = preprocess(init_image).half().cuda()
        

        if method == 'l2':
                adv_X = encoder_pgd_l2(X,
                                model = pipe.vae.encode,
                                clamp_min=-1,
                                clamp_max=1,
                                eps=16/255*math.sqrt(512*512*3),
                                step_size=4/255*math.sqrt(512*512*3),
                                iters=200,
                                experiment=experiment)
        elif method == 'fgsm':
            adv_X = encoder_fgsm(X,
                            model = pipe.vae.encode,
                            clamp_min=-1,
                            clamp_max=1,
                            eps=16/255,
                            step_size=4/255,
                            iters=200,
                            experiment=experiment)
        elif method == 'l1':
            adv_X = encoder_pgd_l1(X,
                            model = pipe.vae.encode,
                            clamp_min=-1,
                            clamp_max=1,
                            eps=16/255*(512*512*3),
                            step_size=4/255*(512*512*3),
                            iters=200,
                            experiment=experiment)
        elif method == 'linf':
            adv_X = encoder_pgd_inf(X,
                            model = pipe.vae.encode,
                            clamp_min=-1,
                            clamp_max=1,
                            eps=16/255,
                            step_size=4/255,
                            iters=200,
                            experiment=experiment)
        elif method == 'random':
            adv_X = X + torch.randn(X.shape).to(X.device) * 16 / 255
        elif method == 'diffusion_2':
            image = prepare_image(init_image)
            image = image.half().cuda()
            target_image_tensor = torch.zeros((512,512)).half().cuda()
            strength = 0.7
            guidance_scale = 7.5
            num_inference_steps = 4
            result, last_image= diffusion_l2(image,
                    prompt=prompt,
                    target_image=target_image_tensor,
                    eps=16/255 * math.sqrt(512*512*3),
                    step_size=4/255 * math.sqrt(512*512*3),
                    iters=200,
                    clamp_min = -1,
                    clamp_max = 1,
                    eta=1,
                    pipe=pipe,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    grad_reps=10)
            adv_X = result
        elif method == 'diffusion_inf':
            image = prepare_image(init_image)
            image = image.half().cuda()
            target_image_tensor = torch.zeros((512,512)).half().cuda()
            strength = 0.7
            guidance_scale = 7.5
            num_inference_steps = 4
            result, last_image= diffusion_linf(image,
                              prompt=prompt,
                              target_image=target_image_tensor,
                              eps=16/255,
                              step_size=4/255,
                              iters=200,
                              clamp_min = -1,
                              clamp_max = 1,
                              height = 512,
                              width = 512,
                              eta=1,
                              pipe=pipe,
                              num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              experiment=experiment
                            )
            adv_X = result
        
        adv_X = (adv_X / 2 + 0.5).clamp(0,1)
    
    adv_image = to_pil(adv_X[0]).convert("RGB")

    GUIDANCE = 7.5
    NUM_STEPS = 50
    STRENGTH = 0.7

    with torch.autocast('cuda'):
        torch.manual_seed(SEED)
        image_nat = pipe(prompt=prompt, strength=STRENGTH, image=init_image, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
        torch.manual_seed(SEED)
        image_adv = pipe(prompt=prompt, strength=STRENGTH, image=adv_image, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
    
    display_images(init_image, adv_image, image_nat, image_adv, prompt, SEED)

    # img1 = T.ToTensor()(image_nat).unsqueeze(0)
    # img2 = T.ToTensor()(image_adv).unsqueeze(0)
    # ssim_score = piq.ssim(img1, img2, data_range=1.0, reduction='mean')
    # print(ssim_score)






def prompts_and_seeds():

    prompt1="Let it be snowy"
    prompt2="Change the background to a beach"
    prompt3="Add a city skyline background"
    prompt4="Add a forest background"
    prompt5="Change the background to a desert"
    prompt6="Set the person stand in a library"
    prompt7="Let the person stand under the moon"
    prompt8="Let the person wear a police suit"
    prompt9="Let the person wear a bowite"
    prompt10="Let the person wear sunglasses"
    prompt11="Let the person wear earrings"
    prompt12="Let the person smoke a cigar"
    prompt13="Place a headband in the hair"
    prompt14="Place a tiara on the top of the head"
    prompt15="Turn the person's hair pink"
    prompt16="Let the person turn bald"
    prompt17="Let the person have a tattoo"
    prompt18="Let the person wear purple makeup"
    prompt19="Let the person grow a mustache"
    prompt20="Turn the person into a zombie"
    prompt21="Change the skin color to Avatar blue"
    prompt22="Add elf-like ears"
    prompt23="Add large vampire fangs"
    prompt24="Apply Goth style makeup"
    prompt25="Make the person smile"
    prompt26="Make the person look like a mermaid"
    prompt27="Add a sad expression"
    prompt28="Make the person look angry"
    prompt29="Add a surprised face"
    prompt30="Turn the face into a wink"
    prompt31="Add a beard"
    prompt32="Make the hair curly"
    prompt33="Change the hairstyle to a bob cut"
    prompt34="Make the person bald"
    prompt35="Add long staight hair"
    prompt36="Add a cowboy hat"
    prompt37="Put on red lipstick"
    prompt38="Add heavy eye shadow"
    prompt39="Make the person look older"
    prompt40="Make the person look younger"
    prompt41="Transfrom the person into a child"
    prompt42="Turn the face into a cartoon style"
    prompt43="Make the face look like an anime character"
    prompt44="Turn the person into cyborg"
    prompt45="Apply Pixar-style redering"
    prompt46="Make the person look like a superhero"
    prompt47="Add a medieval knight helmet"
    prompt48="Dress the person like a doctor"
    prompt49="Make the person look like an astronaut"
    prompt50="Add wrinkles to the face"



    prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10, prompt11, prompt12,
               prompt13, prompt14, prompt15, prompt16, prompt17, prompt18, prompt19, prompt20, prompt21, prompt22, prompt23, prompt24,
               prompt25, prompt26, prompt27, prompt28, prompt29, prompt30, prompt31, prompt32, prompt33, prompt34, prompt35, prompt36,
               prompt37, prompt38, prompt39, prompt40, prompt41, prompt42, prompt43, prompt44, prompt45, prompt46, prompt47, prompt48,]
    SEEDS = [9999, 214, 4971, 4545, 45243, 48901, 5345, 6734, 2342, 2344, 345345, 23423, 12, 23234, 890, 2318, 1231, 123, 52342, 17890,
             73642, 8239, 219, 90, 987, 2736, 34, 23423, 8659, 86432, 32409, 23444, 7777, 77777, 98712, 87654, 12111, 3249, 234, 980,
             8342, 23489, 98011, 2342, 83723, 234, 12394, 98239]
    
    return prompts, SEEDS





def encoder_attack_multiple(method, project_name=None, path=None):
    """Encoder attack and diffusion attack on multiple images."""
    login(token=TOKEN)

    to_pil = T.ToPILImage()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    init_images = load_images(folder=path)
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    init_images = [center_crop(resize(init_image)) for init_image in init_images]


    prompts, SEEDS = prompts_and_seeds()

    ssim_scores = []
    psnr_scores = []
    msssim_scores = []
    iwssim_scores = []
    vifp_scores = []
    fsim_scores = []
    images1 = []
    images2 = []

    for i in tqdm(range(len(SEEDS))):

        if project_name:
            experiment = Experiment(project_name=PROJECT_NAME,)
            experiment.set_name(f"Experiment {i}")
        else:
            experiment = None

        with torch.autocast('cuda'):
            X = preprocess(init_images[i]).half().cuda()
            if method == 'l2':
                adv_X = encoder_pgd_l2(X,
                                model = pipe.vae.encode,
                                clamp_min=-1,
                                clamp_max=1,
                                eps=16/255*math.sqrt(512*512*3),
                                step_size=4/255*math.sqrt(512*512*3),
                                iters=200,
                                experiment=experiment)
            elif method == 'fgsm':
                adv_X = encoder_fgsm(X,
                            model = pipe.vae.encode,
                            clamp_min=-1,
                            clamp_max=1,
                            eps=16/255,
                            step_size=4/255,
                            iters=200,
                            experiment=experiment)
            elif method == 'l1':
                adv_X = encoder_pgd_l1(X,
                            model = pipe.vae.encode,
                            clamp_min=-1,
                            clamp_max=1,
                            eps=16/255*(512*512*3),
                            step_size=4/255*(512*512*3),
                            iters=200,
                            experiment=experiment)
            elif method == 'linf':
                adv_X = encoder_pgd_inf(X,
                            model = pipe.vae.encode,
                            clamp_min=-1,
                            clamp_max=1,
                            eps=16/255,
                            step_size=4/255,
                            iters=200,
                            experiment=experiment)
            elif method == 'random':
                adv_X = X + torch.randn(X.shape).to(X.device) * 16 / 255
            elif method == 'diffusion_2':
                image = prepare_image(init_images[i])
                image = image.half().cuda()
                target_image_tensor = torch.zeros((512,512)).half().cuda()
                strength = 0.7
                guidance_scale = 7.5
                num_inference_steps = 4
                result, last_image= diffusion_l2(image,
                    prompt=prompts[i],
                    target_image=target_image_tensor,
                    eps=16/255 * math.sqrt(512*512*3),
                    step_size=4/255 * math.sqrt(512*512*3),
                    iters=200,
                    clamp_min = -1,
                    clamp_max = 1,
                    eta=1,
                    pipe=pipe,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    grad_reps=10)
                adv_X = result
            elif method == 'diffusion_inf':
                image = prepare_image(init_images[i])
                image = image.half().cuda()
                target_image_tensor = torch.zeros((512,512)).half().cuda()
                strength = 0.7
                guidance_scale = 7.5
                num_inference_steps = 4
                result, last_image= diffusion_linf(image,
                              prompt=prompts[i],
                              target_image=target_image_tensor,
                              eps=16/255,
                              step_size=4/255,
                              iters=200,
                              clamp_min = -1,
                              clamp_max = 1,
                              height = 512,
                              width = 512,
                              eta=1,
                              pipe=pipe,
                              num_inference_steps=num_inference_steps,
                              guidance_scale=guidance_scale,
                              experiment=experiment
                            )
                adv_X = result
            
        
            adv_X = (adv_X / 2 + 0.5).clamp(0,1)
    
        adv_image = to_pil(adv_X[0]).convert("RGB")

        GUIDANCE = 7.5
        NUM_STEPS = 50
        STRENGTH = 0.7

        with torch.autocast('cuda'):
            torch.manual_seed(SEEDS[i])
            image_nat = pipe(prompt=prompts[i], strength=STRENGTH, image=init_images[i], guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
            torch.manual_seed(SEEDS[i])
            image_adv = pipe(prompt=prompts[i], strength=STRENGTH, image=adv_image, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
    
        # display_images(init_images[i], adv_image, image_nat, image_adv, prompts[i], SEEDS[i])

        if experiment:
            experiment.log_image(init_images[i], name="init_image")
            experiment.log_image(adv_image, name="adv_image")
            experiment.log_image(image_nat, name="image_nat")
            experiment.log_image(image_adv, name="image_adv")
            img1 = T.ToTensor()(image_nat).unsqueeze(0)
            img2 = T.ToTensor()(image_adv).unsqueeze(0)

            ssim_score = piq.ssim(img1, img2, data_range=1.0)
            experiment.log_metric("SSIM", ssim_score)
            ssim_scores.append(ssim_score)

            psnr_score = piq.psnr(img1, img2, data_range=1.0)
            experiment.log_metric("PSNR", psnr_score)
            psnr_scores.append(psnr_score)

            vifp_score = piq.vif_p(img1, img2, data_range=1.0)
            experiment.log_metric("VIFp", vifp_score)
            vifp_scores.append(vifp_score)

            fsim_score = piq.fsim(img1, img2, data_range=1.0)
            experiment.log_metric("FSIM", fsim_score)
            fsim_scores.append(fsim_score)

            images1.append(img1)
            images2.append(img2)
    
    if experiment:

        ssim_scores = [score.item() for score in ssim_scores]
        psnr_scores = [score.item() for score in psnr_scores]
        vifp_scores = [score.item() for score in vifp_scores]
        fsim_scores = [score.item() for score in fsim_scores]


        ssim_mean = statistics.mean(ssim_scores)
        psnr_mean = statistics.mean(psnr_scores)
        vifp_mean = statistics.mean(vifp_scores)
        fsim_mean = statistics.mean(fsim_scores)

        ssim_std = statistics.stdev(ssim_scores)
        psnr_std = statistics.stdev(psnr_scores)
        vifp_std = statistics.stdev(vifp_scores)
        fsim_std = statistics.stdev(fsim_scores)

        experiment.log_metric("SSIM mean", ssim_mean)
        experiment.log_metric("PSNR mean", psnr_mean)
        experiment.log_metric("vifp mean", vifp_mean)
        experiment.log_metric("fsim mean", fsim_mean)
        experiment.log_metric("SSIM std", ssim_std)
        experiment.log_metric("PSNR std", psnr_std)
        experiment.log_metric("vifp std", vifp_std)
        experiment.log_metric("fsim std", fsim_std)



        tensor_images1 = [img.squeeze(0) for img in images1]  # Remove the leading dimension of 1
        tensor_images2 = [img.squeeze(0) for img in images2]


        # Create a custom dataset that returns dictionaries with 'images' key
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, images):
                self.images = images
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return {'images': self.images[idx]}

        # Create datasets
        dataset1 = DictDataset(tensor_images1)
        dataset2 = DictDataset(tensor_images2)

        # Define a collate function that maintains the dictionary structure
        def dict_collate(batch):
            return {
                'images': torch.stack([item['images'] for item in batch])
            }

        # Create dataloaders with our custom collate function
        dataloader1 = torch.utils.data.DataLoader(
            dataset1, 
            batch_size=4, 
            shuffle=False,
            collate_fn=dict_collate
        )

        dataloader2 = torch.utils.data.DataLoader(
            dataset2, 
            batch_size=4, 
            shuffle=False,
            collate_fn=dict_collate
        )

        # For debugging, check the shape after processing
        for batch in dataloader1:
            print(f"Batch['images'] shape: {batch['images'].shape}")
            break

        # Calculate FID and PR scores
        fid_metric = piq.FID()
        pr_metric = piq.PR()

        # Compute features
        first_feats = fid_metric.compute_feats(dataloader1)
        second_feats = fid_metric.compute_feats(dataloader2)

        # Calculate metrics
        fid_score = fid_metric(first_feats, second_feats)
        pr_score = pr_metric(first_feats, second_feats)
        experiment.log_metric("FID", fid_score)
        experiment.log_metric("PR", pr_score)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", 
                        choices=['l1', 'l2', 'linf', 'random', 'pgsm', 'diffusion_2', 'diffusion_inf'],
                        default='linf')
                        #help="Attack method. Choices: 'l1', 'l2', 'linf', 'random', 'pgsm', 'diffusion_2', 'diffusion_inf'.")
    parser.add_argument("--batch",
                        action='store_true',
                        help="Enable attack on multiple images (default: False, no argument required).")
    parser.add_argument("--path", default=None,
                        help="image or image folder path.")
    args = parser.parse_args()

    path = args.path
    method = args.method
    batch = args.batch


    if batch:
        encoder_attack_multiple(method=method, path=path, project_name=PROJECT_NAME)
    else:
        encoder_attack(method=method, path=path, project_name=PROJECT_NAME)




