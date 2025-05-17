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
from torch import nn
from torchvision import models
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



def early_fusion(X, model, model_res, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mu=(1,1), experiment=None):
    "Early Fusion joint attack."

    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        #actual_step_size = step_size
        X_adv.requires_grad_(True)
        loss_enc = (model(X_adv).latent_dist.mean).norm()


        loss_func = nn.CrossEntropyLoss()
        original_logit = model_res(X_adv)
        original_target = original_logit.argmax(dim=1)
        if original_target == 0:
            loss_res = original_logit[0][0]
            llllllll = original_logit[0][1]
        else:
            loss_res = original_logit[0][1]
            llllllll = original_logit[0][0]
        
        # Debugging line
        # print(loss_enc, loss_res, llllllll, loss_enc / loss_res)
        
        loss =  loss_enc  -  mu[0] * llllllll + mu[1] * loss_enc



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




def load_model(model_path):
    DEVICE = torch.device('cuda')
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (male, female)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def prob(num1, num2):
    return math.exp(num1) / (math.exp(num1) + math.exp(num2))



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






def joint_attack(path=None, eps=16/255*math.sqrt(512*512*3), step_size=2/255*math.sqrt(512*512*3), iters=200, mu=(1,1), project_name=None):
    """Joint Attack."""
    
    login(token=TOKEN)

    to_pil = T.ToPILImage()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    init_images = load_images(path)
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    init_images = [center_crop(resize(init_image)) for init_image in init_images]


    folder = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(folder,'model.pth')

    num = 0
    model = load_model(MODEL_PATH)
    new_list = []
    original_list = []


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
            adv_X = early_fusion(X,
                            model = pipe.vae.encode,
                            model_res = model,
                            clamp_min=-1,
                            clamp_max=1,
                            eps=eps,
                            step_size=step_size,
                            iters=iters,
                            mu=mu,
                            experiment=experiment)
            





            with torch.no_grad():
                original_logit = model(X)
                new_logit = model(adv_X)
                original = original_logit.argmax(dim=1)
                new  = new_logit.argmax(dim=1)
        
                if original[0] == 0:
                    original_list.append(prob(original_logit[0][0].item(), original_logit[0][1].item()))
                else:
                    original_list.append(prob(original_logit[0][1].item(), original_logit[0][0].item()))
                
                if new[0] == 0:
                    new_list.append(prob(new_logit[0][0].item(), new_logit[0][1].item()))
                else:
                    new_list.append(prob(new_logit[0][1].item(), new_logit[0][0].item()))

            
                if original != new:
                    num += 1
            
            # Debugging line
            # print("current number",num)





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

        



        original_mean = statistics.mean(original_list)
        new_mean = statistics.mean(new_list)
        
        original_std = statistics.stdev(original_list)
        new_std = statistics.stdev(new_list)

        experiment.log_metric("success attack", num)
        experiment.log_metric("original mean", original_mean)
        experiment.log_metric("new mean ", new_mean)
        experiment.log_metric("original std", original_std)
        experiment.log_metric("new std", new_std)




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
        # for batch in dataloader1:
        #     print(f"Batch['images'] shape: {batch['images'].shape}")
        #     break

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
    parser.add_argument("--path", default=None,
                        help="image or image folder path.")
    parser.add_argument("--eps", type=float, default=16/255*math.sqrt(512*512*3),
                        help="attack budget.")
    parser.add_argument("--step_size", type=float, default=2/255*math.sqrt(512*512*3),
                        help="PGD initial step size.")
    parser.add_argument("--iters", type=int, default=200,
                        help="num of PGD iterations.")
    parser.add_argument("--mu", type=float, nargs=2,
                        metavar=('mu_1', 'mu_2'),
                        default=(1,1),
                        help="loss function scaling hyperparameters.")
    args = parser.parse_args()
    path = args.path
    eps = args.eps
    step_size = args.step_size
    iters = args.iters
    mu = args.mu
    project_name = PROJECT_NAME


    joint_attack(path, eps, step_size, iters, mu, project_name)



