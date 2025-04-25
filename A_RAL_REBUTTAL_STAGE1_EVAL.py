import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import torch,clip
import torch.nn as nn
from traj_predict.model.transformer_for_diffusion import TransformerForDiffusion 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import models.vision_transformer as vits
from traj_predict.Real_Robot_Stage1_Dataloader import LMDBDataset,DataPrefetcher
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, RandomResizedCrop 
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_scheduler
import wandb
from time import time
import matplotlib.pyplot as plt
from traj_predict.traj_func import Real_Robot_2D_PreProcess,resize_points
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

def normalize_data(data, stats={'min': 0,'max': 224}):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata
def unnormalize_data(ndata, stats={'min': 0,'max': 224}):
    ndata = (ndata + 1) / 2 # [-1, 1] -> [0, 1] 域
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


def transform_points(points, crop_box, transformed_image):
    transformed_points = []
    for batch_idx in range(points.shape[0]):
        for seq_idx in range(points.shape[1]):
            crop_x, crop_y, crop_w, crop_h = crop_box[batch_idx, seq_idx]
            scale_x = transformed_image.shape[-1] / crop_w
            scale_y = transformed_image.shape[-2] / crop_h
            transformed_points.append([(int((x - crop_x) * scale_x), int((y - crop_y) * scale_y)) for x, y in points[batch_idx, seq_idx]])
    transformed_points = torch.tensor(transformed_points).unsqueeze(1)
    return transformed_points
    
        
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


if __name__ == '__main__':
   
    # config prepare
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 96
    num_workers = 4
    # lmdb_dir = "/bamboo_dir/24101516_pick_bread_plate_1/success_episodes_lmdb_1/"
    lmdb_dir = "/media/users/bamboo/dataset/lmdb/ral_rebuttal_2/"
    #image preprocess
    preprocessor = Real_Robot_2D_PreProcess(
        rgb_static_pad = 10, # 去除位置敏感性
        rgb_shape = [224,224], 
        rgb_mean = [0.485, 0.456, 0.406],
        rgb_std =  [0.229, 0.224, 0.225],
        device = device
    )
    # data loader
    train_dataset = LMDBDataset(
        lmdb_dir = lmdb_dir, 
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过30
        action_dim = 2, # x,y,gripper_state
        start_ratio = 0,
        end_ratio = 0.95, 
    )
    val_dataset = LMDBDataset(
        lmdb_dir = lmdb_dir, 
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过30
        action_dim = 2,
        start_ratio = 0.95,
        end_ratio = 1, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, # to be flattened in prefetcher  
        num_workers=num_workers,
        pin_memory=True, # Accelerate data reading
        shuffle=False,
        prefetch_factor=2,
        persistent_workers=True,
    ) 
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, # to be flattened in prefetcher  
        num_workers=num_workers,
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
    ) 
    train_prefetcher = DataPrefetcher(train_loader, device)
    val_prefetcher = DataPrefetcher(val_loader, device)
             
    model = TrajPredictPolicy()
    # 预训练模型读入
    model_path = "Save/RAL_Rebuttal_stage1.pth"
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
    
    # policy config
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
    val_total_loss = 0
    val_index = 0
    with tqdm(total=len(val_loader), desc=f"Val Epoch {72}", leave=False) as pbar:
            with torch.no_grad():
                batch, load_time = val_prefetcher.next()
                val_index = 0
                while batch is not None and val_index < 20000:
                    eval_flag = True

                    if eval_flag:
                        model.eval()
                        language = batch['inst_token']
                        image = batch['rgb_camera']
                        naction = batch['actions']
                        # example inputs
                        rgb_top_norm,naction_transformed = preprocessor.rgb_process(batch['rgb_camera'],batch['actions'],train=False)    
                        naction_trans_norm = normalize_data(naction_transformed)
                        noisy_action = torch.randn(naction.shape, device=device)
                        batch_val_size,sequence,chunk_size,dim = noisy_action.shape
                        noisy_action = noisy_action.reshape(batch_val_size*sequence,chunk_size,dim)
                        out_action = noisy_action
                        # init scheduler
                        noise_scheduler.set_timesteps(num_diffusion_iters)
                        language_embedding, obs_embeddings, patch_embeddings = None, None, None
                        # val_sample_loss_values = []
                        start_time = time()

                        for k in noise_scheduler.timesteps:
                            # predict noise
                            noise_pred, language_embedding, obs_embeddings, patch_embeddings = model(rgb_top_norm, language, timesteps=k, noisy_actions=out_action,
                                                language_embedding=language_embedding, obs_embeddings=obs_embeddings, patch_embeddings=patch_embeddings)
                            # inverse diffusion step (remove noise)
                            out_action = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=out_action
                            ).prev_sample
                            # val_sample_loss =nn.functional.mse_loss(out_action, naction_trans_norm.squeeze(1))
                            # val_sample_loss_values.append(val_sample_loss.cpu())
                        end_time = time()
                        execution_time = end_time - start_time


                        
                        re_out_action = unnormalize_data(out_action)
                        val_sample_loss =nn.functional.mse_loss(out_action, naction_trans_norm.squeeze(1))
                        val_total_loss += val_sample_loss.item()
                        # print(val_sample_loss.item())
                        # visualization croped image
                        # ################Convert tensor to NumPy array for visualization
                        re_out_action = re_out_action.unsqueeze(1)
                        re_out_action_ori = resize_points(re_out_action.clone(), (224,224), (640,640))
                        # re_out_action_ori = (re_out_action - torch.tensor([0, 80]).to(device))/torch.tensor([0.5, 0.667]).to(device)
                
                        import cv2
                        for batch_idx in range(image.shape[0]):
                            for seq_idx in range(image.shape[1]):
                                rgb_camera = image[batch_idx][seq_idx][:, 80:-80, :].permute(1, 2, 0).cpu().numpy().copy()
                                for index, point_2d in enumerate(naction[batch_idx,seq_idx,:,:] - torch.tensor([0, 80]).to(device)):
                                    color = (
                                        int(255 * (index / naction.shape[2])),  # 红色分量
                                        int(206 * (index / naction.shape[2])),  # 绿色分量
                                        int(135 * (index / naction.shape[2]))   # 蓝色分量
                                    )
                                    cv2.circle(rgb_camera, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
                                    cv2.circle(rgb_camera, tuple(point_2d.int().tolist()), radius=6, color=color, thickness=-1)
                                cv2.putText(rgb_camera, batch["inst"][batch_idx], (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
                                cv2.imwrite(f"visualization/rgb_camera_groundtruth{val_index}_{batch_idx}.png", rgb_camera)  
                                
                                rgb_camera = image[batch_idx][seq_idx][:, 80:-80, :].permute(1, 2, 0).cpu().numpy().copy()
                                for index, point_2d in enumerate(re_out_action_ori[batch_idx,seq_idx,:,:] - torch.tensor([0, 80]).to(device)):
                                    color = (
                                        int(255 * (index / re_out_action_ori.shape[2])),  # 红色分量
                                        int(206 * (index / re_out_action_ori.shape[2])),  # 绿色分量
                                        int(135 * (index / re_out_action_ori.shape[2]))   # 蓝色分量
                                    )
                                    cv2.circle(rgb_camera, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
                                    cv2.circle(rgb_camera, tuple(point_2d.int().tolist()), radius=6, color=color, thickness=-1)
                                    # cv2.circle(rgb_camera_top, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                                
                                cv2.imwrite(f"visualization/rgb_camera_groundtruth_pred{val_index}_{batch_idx}.png", rgb_camera)  

                                # rgb_top_reshape = preprocessor.rgb_recovery(rgb_top_norm)
                                # rgb_top_np = rgb_top_reshape[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy().copy()
                                # for point_2d in naction_transformed[batch_idx,seq_idx,:,:]:
                                #     cv2.circle(rgb_top_np, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                                # cv2.imwrite("traj_predict/script/visualization/rgb_camera_top_trans.png", rgb_top_np)  
            
                                # rgb_top_np2 = rgb_top_reshape[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy().copy()
                                # for point_2d in re_out_action[batch_idx,seq_idx,:,:]:
                                #     cv2.circle(rgb_top_np2, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                                # cv2.imwrite("traj_predict/script/visualization/rgb_camera_top_preds.png", rgb_top_np2)                  
                                # cv2.waitKey(1)
                        val_index = val_index + 1
                    batch, load_time = val_prefetcher.next()

                    pbar.update(1)
            avg_val_loss = val_total_loss/val_index
            print(f"avg_val_loss: {avg_val_loss}")
       




