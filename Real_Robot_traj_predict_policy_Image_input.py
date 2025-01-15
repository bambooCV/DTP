import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import torch,clip
import torch.nn as nn
from traj_predict.model.transformer_for_diffusion import TransformerForDiffusion 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import models.vision_transformer as vits
from traj_predict.Real_Robot_traj_diffusion_data_loader import LMDBDataset,DataPrefetcher,contains_words
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, RandomResizedCrop 
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_scheduler

from time import time
import matplotlib.pyplot as plt
from traj_predict.traj_func import Real_Robot_2D_PreProcess,resize_points
import cv2
from torchvision.transforms.v2 import Resize


# 定义颜色范围（浅色和深色，BGR 格式）
color_start = np.array([13, 0, 103])      # 深色 (BGR: 103, 0, 13)
color_end = np.array([240, 245, 255])  # 浅色 (BGR: 255, 245, 240)

def interpolate_color(index, total_points, color_start, color_end):
    """根据 index 插值颜色，从浅到深渐变"""
    ratio = index / total_points  # 当前点的比例
    color = color_start + (color_end - color_start) * ratio
    return tuple(map(int, color))  # 转换为 Python 原生整数元组

def unnormalize_data(ndata, stats={'min': 0,'max': 224}):
    ndata = (ndata + 1) / 2 # [-1, 1] -> [0, 1] 域
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
class TrajPredictPolicy(nn.Module):
    def __init__(
        self,
        
    ):
        super().__init__()

        # vision encoders model
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        checkpoint_vit = torch.load("/media/users/bamboo/PretrainModel/vit/mae_pretrain_vit_base.pth")
        model_mae.load_state_dict(checkpoint_vit['model'], strict=False)
        # language encoders model
        model_clip, _ = clip.load("/media/users/bamboo/PretrainModel/clip/ViT-B-32.pt",device="cpu") 
       
        # CLIP for language encoding
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False
        # MAE for image encoding
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
            
            
        self.hidden_size = 512
        self.img_feat_dim = 768
        # project functions for images
        self.proj_static_img = torch.nn.Linear(self.img_feat_dim, self.hidden_size)
        
        # predict noise model 
        self.action_dim = 2 # x,y
        self.action_horizon = 30
        self.patch_size = 14
        self.noise_pred_net =  TransformerForDiffusion(
                input_dim=self.action_dim ,
                output_dim=self.action_dim ,
                horizon=self.action_horizon,
                n_obs_steps=1*self.patch_size**2,
                cond_dim=512,
                causal_attn=True,
                # time_as_cond=False,
                # n_cond_layers=4
            )
              
    def forward(self, 
        rgb_static_norm,
        language,
        timesteps,
        noisy_actions,
        language_embedding = None,
        obs_embeddings = None,
        patch_embeddings = None
        ):
        # model input prepare: noisy_actions, timesteps, obs_cond
        # image batch*seq, channel, height, width
        batch_size, sequence, channel, height, width = rgb_static_norm.shape
        rgb_static_norm = rgb_static_norm.view(batch_size*sequence, channel, height, width)

        if language_embedding is None and obs_embeddings is None and patch_embeddings is None:
            with torch.no_grad():
                language_embedding = self.model_clip.encode_text(language).unsqueeze(1)
                obs_embeddings, patch_embeddings = self.model_mae(rgb_static_norm)
            patch_embeddings = self.proj_static_img(patch_embeddings)

        
        # concatenate vision feature and language obs
        obs_features = torch.cat([patch_embeddings, language_embedding], dim=1)
        obs_cond = obs_features
        
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, obs_cond)
        
        return noise_pred,language_embedding,obs_embeddings,patch_embeddings


# preprocess
resize = Resize([224,224], interpolation=Image.BICUBIC, antialias=True)
rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) 
rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajPredictPolicy()
# 预训练模型读入
# model_path = "Save/Real_Robot_2D_evaluation_0114.pth"
model_path = "Save/Real_Robot_2D_evaluation_hflip.pth"
state_dict = torch.load(model_path,map_location=device)['model_state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')
    new_state_dict[new_key] = value
    multi_gpu = True
if multi_gpu:
    model.load_state_dict(new_state_dict,strict=False)
else:
    model.load_state_dict(state_dict,strict=False)
model = model.to(device)


num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)


with torch.no_grad():
    model.eval()
    # language = "put the yellow pepper and place it into bowl"
    # language = "put the red pepper and place it into bowl"
    # language = "put the yellow pepper and place it into basket"
    # language = "put the red pepper and place it into basket"
    language = "grasp brown steamed buns in the pan"
    # language = "close the pot"
    tokenizer = clip.tokenize
    tokenized_text = tokenizer(language).to(device)
    # image = cv2.imread("/home/bamboofan/EmbodiedAI/multiview_dataaug/Grounded-Segment-Anything/A_visualization/pick_bread_ori.jpg")
    # image = cv2.imread("/home/bamboofan/EmbodiedAI/multiview_dataaug/Grounded-Segment-Anything/A_visualization/remove/pick_bread_aug_280_234.jpg")
    image = cv2.imread("/media/users/bamboo/EmbodiedAI/DTP/tools/visualization/h5_vis.png")
    image = cv2.resize(image, dsize=(640,480))
    # image 转 tensor
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    image_tensor = F.pad(
                            image_tensor,
                            pad=(0, 0, 80, 80),  # 在高度方向填充
                            mode='replicate'  # 使用边缘像素填充
                        )
    rgb_camera_top = image_tensor.float()*(1/255.)
    rgb_camera_top = resize(rgb_camera_top)
    rgb_top_norm = (rgb_camera_top - rgb_mean) / (rgb_std + 1e-6)
    rgb_top_norm = rgb_top_norm.unsqueeze(0).unsqueeze(0).to(device)
    
    noisy_action = torch.randn([1,30,2], device=device)
    out_action = noisy_action
    # init scheduler
    noise_scheduler.set_timesteps(num_diffusion_iters)
    language_embedding, obs_embeddings, patch_embeddings = None, None, None


    for k in noise_scheduler.timesteps:
        # predict noise
        noise_pred, language_embedding, obs_embeddings, patch_embeddings = model(rgb_top_norm, tokenized_text, timesteps=k, noisy_actions=out_action,
                            language_embedding=language_embedding, obs_embeddings=obs_embeddings, patch_embeddings=patch_embeddings)
        # inverse diffusion step (remove noise)
        out_action = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=out_action
        ).prev_sample

    re_out_action = unnormalize_data(out_action)


    re_out_action = re_out_action.unsqueeze(1)
    re_out_action_ori = resize_points(re_out_action.clone(), (224,224), (640,640))
    re_out_action_ori = re_out_action_ori.squeeze(0).squeeze(0)



    rgb_vis = image_tensor.permute(1, 2, 0).numpy().copy()
    for index, point_2d in enumerate(re_out_action_ori):
        color = (
            int(255 * (index / 30)),  # 红色分量
            int(206 * (index / 30)),  # 绿色分量
            int(135 * (index / 30))   # 蓝色分量
        )
        cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
        cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=6, color=color, thickness=-1)
        # cv2.circle(rgb_camera_top, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
    
    cv2.imwrite(f"tools/visualization/diffusion_inference.png", rgb_vis)  
    rgb_vis = image_tensor.permute(1, 2, 0).numpy().copy()
    # 绘制点
    for index, point_2d in enumerate(re_out_action_ori):
        # 获取渐变颜色
        color = interpolate_color(index, len(re_out_action_ori) - 1, color_start, color_end)
        
        # 绘制外圈白色圆圈
        cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
        
        # 绘制内圈渐变色圆圈
        cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=6, color=color, thickness=-1)

    # 保存图像
    cv2.imwrite("tools/visualization/diffusion_inference_gradient.png", rgb_vis)