import warnings

import os
import sys
sys.path.append('..')
# sys.path.append('/home/ps/Dev/inrocs/')
import threading
import time
from pathlib import Path
from einops import rearrange

import torch
import cv2
import json

import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# from pynput import keyboard

from models.real_robot_gr1 import GR1 
# from robot_env.franka_env import robot_env
import models.vision_transformer as vits
import torch.nn.functional as F
from PreProcess import Real_Robot_PreProcess
# from leader.agent import LeaderAgent, BiLeaderAgent
# print('aaa')
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
import clip
import torch.nn as nn
from traj_predict.model.transformer_for_diffusion import TransformerForDiffusion 
preparing = True
def unnormalize_data(ndata, stats={'min': 0,'max': 224}):
    ndata = (ndata + 1) / 2 # [-1, 1] -> [0, 1] 域
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
def resize_points(points, original_size, new_size):
    original_h, original_w = original_size
    new_h, new_w = new_size
    
    # 计算缩放比例
    scale_x = new_w / original_w
    scale_y = new_h / original_h
    
    # 调整点的坐标
    points[..., 0] = points[..., 0] * scale_x
    points[..., 1] = points[..., 1] * scale_y
    
    return points
class TrajPredictPolicy(nn.Module):
    def __init__(
        self,
        
    ):
        super().__init__()

        # vision encoders model
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        checkpoint_vit = torch.load("../Pretrain_Model/vit/mae_pretrain_vit_base.pth")
        model_mae.load_state_dict(checkpoint_vit['model'], strict=False)
        # language encoders model
        model_clip, _ = clip.load("ViT-B/32",device="cpu") 
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

class DTP_Evaluation():
    def __init__(self,cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = Real_Robot_PreProcess(
            cfg['rgb_static_pad'],
            cfg['rgb_shape'],
            cfg['rgb_mean'],
            cfg['rgb_std'],
            self.device,
        )
        # model_clip, _ = clip.load(cfg['clip_backbone']) 
        model_clip, _ = clip.load(cfg['clip_ckpt']) 
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        checkpoint = torch.load(cfg['mae_ckpt'],weights_only=False)
        model_mae.load_state_dict(checkpoint['model'], strict=False)
        training_target = ['act_pred', 'fwd_pred', 'fwd_pred_multiview']
        self.model = GR1(
            model_clip,
            model_mae,
            state_dim=cfg['state_dim'],
            act_dim=cfg['act_dim'],
            hidden_size=cfg['embed_dim'],
            sequence_length=cfg['seq_len'],
            chunk_size=cfg['chunk_size'],
            training_target=training_target,
            img_feat_dim=cfg['img_feat_dim'],
            patch_feat_dim=cfg['patch_feat_dim'],
            lang_feat_dim=cfg['lang_feat_dim'],
            resampler_params={
                'depth': cfg['resampler_depth'],
                'dim_head': cfg['resampler_dim_head'],
                'heads': cfg['resampler_heads'],
                'num_latents': cfg['resampler_num_latents'],
                'num_media_embeds': cfg['resampler_num_media_embeds'],
            },
            without_norm_pixel_loss=False,
            use_multi_rgb=cfg['use_multi_rgb'],
            use_2d_traj = cfg['use_2d_traj'],
            n_layer=cfg['n_layer'],
            n_head=cfg['n_head'],
            n_inner=4*cfg['embed_dim'],
            activation_function=cfg['activation_function'],
            n_positions=cfg['n_positions'],
            resid_pdrop=cfg['dropout'],
            attn_pdrop=cfg['dropout'],
        ).to(self.device)
        state_dict = torch.load(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch']),map_location=self.device)['state_dict'] 
        self.stats = torch.load(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch']))['stats']
        self.model.load_state_dict(state_dict, strict=False)

        # cfg
        self.tokenizer = clip.tokenize
        self.seq_len = cfg['seq_len']
        self.chunk_size = cfg['chunk_size']
        self.test_chunk_size = cfg['test_chunk_size']
        self.use_multi_rgb = cfg['use_multi_rgb']
        self.act_dim = cfg['act_dim']
        self.state_dim = cfg['state_dim']
        self.action_mode = cfg['action_mode']
        self.use_2d_traj = cfg['use_2d_traj']
        if self.use_2d_traj:
        # 2d traj 
            self.policy_traj = TrajPredictPolicy()
            traj_model_path = "Save/Real_Robot_2D_pick_bread_total.pth"
            traj_state_dict = torch.load(traj_model_path,map_location=self.device)['model_state_dict']
            new_state_dict = {}
            for key, value in traj_state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
                multi_gpu = True
            if multi_gpu:
                self.policy_traj.load_state_dict(new_state_dict,strict=False)
            else:
                self.policy_traj.load_state_dict(traj_state_dict,strict=False)
            self.policy_traj = self.policy_traj.to(self.device)

        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        # ensembling
        self.temporal_ensembling = True
        self.max_publish_step = 10000
        self.step = 0
        self.all_time_actions = torch.zeros([self.max_publish_step, self.max_publish_step + self.chunk_size, self.state_dim]).to(self.device)
        self.diff_flag = False
    def reset(self):
        """Reset function."""
        self.rgb_left_list = []
        self.rgb_right_list = []
        self.rgb_top_list = []
        self.state_list = []
        self.traj_2d_list = []
        self.rollout_step_counter = 0
        
        self.step = 0
        self.all_time_actions = torch.zeros([self.max_publish_step, self.max_publish_step + self.chunk_size, self.state_dim]).to(self.device)

    def inference(self,obs):
        # Language
        text = "pick up the bread and put it on the plate"
        tokenized_text = self.tokenizer(text)

        # get images
        all_cam_images =[]
        for cam_name in ['left', 'right', 'top']:

            cam_img = obs['images'][cam_name]
            cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)

            if cam_name == 'top':
                cam_img = cv2.resize(cam_img, dsize=(640,480))  # (480, 640, 3)
            
            padding_value = (640 - 480) // 2  # 假设宽度大于高度
            cam_img = cv2.copyMakeBorder(
                cam_img,
                top=padding_value, bottom=padding_value, left=0, right=0,  # 只在高度方向填充
                borderType=cv2.BORDER_REPLICATE  # 使用边缘填充
            )

            # cv2.imwrite(f"visualization/cam_img_padding_{cam_name}.png",cam_img)
            all_cam_images.append(cam_img)
        
        # RGB
        rgb_left = rearrange(torch.from_numpy(all_cam_images[0]), 'h w c -> c h w')
        rgb_right = rearrange(torch.from_numpy(all_cam_images[1]), 'h w c -> c h w')
        rgb_top = rearrange(torch.from_numpy(all_cam_images[2]), 'h w c -> c h w')
        self.rgb_left_list.append(rgb_left)
        self.rgb_right_list.append(rgb_right)
        self.rgb_top_list.append(rgb_top)

        # State
        arm_state = obs['qpos'][:7]
        # arm_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        gripper_state = 0 if obs['qpos'][-1] < 0.2 else 1
        state = torch.from_numpy(np.hstack([arm_state, gripper_state]))
        self.state_list.append(state)

        # Buffer
        buffer_len = len(self.rgb_left_list)
        if buffer_len > self.seq_len:
            self.rgb_left_list.pop(0)
            self.rgb_right_list.pop(0)
            self.rgb_top_list.pop(0)
            self.state_list.pop(0)
            assert len(self.rgb_left_list) == self.seq_len
            assert len(self.rgb_right_list) == self.seq_len
            assert len(self.rgb_top_list) == self.seq_len
            assert len(self.state_list) == self.seq_len
            buffer_len = len(self.rgb_left_list)
        
        # left RGB
        c, h, w = rgb_left.shape
        rgb_left_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_left_tensor = torch.stack(self.rgb_left_list, dim=0)  # (t, c, h, w)
        rgb_left_data[0, :buffer_len] = rgb_left_tensor

        # right RGB
        c, h, w = rgb_right.shape
        rgb_right_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_right_tensor = torch.stack(self.rgb_right_list, dim=0)  # (t, c, h, w)
        rgb_right_data[0, :buffer_len] = rgb_right_tensor

        # top RGB
        c, h, w = rgb_top.shape
        rgb_top_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_top_tensor = torch.stack(self.rgb_top_list, dim=0)  # (t, c, h, w)
        rgb_top_data[0, :buffer_len] = rgb_top_tensor

        # State
        state_tensor = torch.stack(self.state_list, dim=0)  # (l, state_dim)
        gripper_state_data =  torch.zeros((1, self.seq_len)).float()
        gripper_state_data[0, :buffer_len] = state_tensor[:, -1]
        gripper_state_data = torch.where(gripper_state_data < 0.2, torch.tensor(0.0), torch.tensor(1.0)).long()
        gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).float()  # (1, t, 2)
        arm_state_data = torch.zeros((1, self.seq_len, self.state_dim - 1)).float()  # (1, t, state_dim - 1)
        arm_state_data[0, :buffer_len] = state_tensor[:, :self.state_dim - 1]

        # Attention mask
        attention_mask = torch.zeros(1, self.seq_len).long()
        attention_mask[0, :buffer_len] = 1
        
        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        rgb_left_data = rgb_left_data.to(self.device)
        rgb_right_data = rgb_right_data.to(self.device)
        rgb_top_data = rgb_top_data.to(self.device)
        arm_state_data = arm_state_data.to(self.device)
        gripper_state_data = gripper_state_data.to(self.device)
        state_data = {'arm': arm_state_data, 'gripper': gripper_state_data}
        attention_mask = attention_mask.to(self.device)
        rgb_left_data, rgb_right_data, rgb_top_data = self.preprocessor.rgb_process(rgb_left_data, rgb_right_data, rgb_top_data, train=False)
        if self.use_2d_traj:
            # DTM inference
            if self.rollout_step_counter == 0 or self.diff_flag == True:
                print("Diffusion trajectory generation")
                with torch.no_grad():
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
                    rgb_top_norm = rgb_top_data[:,len(self.rgb_top_list)-1].unsqueeze(0)
                    noisy_action = torch.randn([1,30,2], device=self.device)
                    out_action = noisy_action
                    language_embedding, obs_embeddings, patch_embeddings = None, None, None

                    for k in self.noise_scheduler.timesteps:
                            # predict noise
                            noise_pred, language_embedding, obs_embeddings, patch_embeddings = self.policy_traj(rgb_top_norm, tokenized_text, timesteps=k, noisy_actions=out_action,
                                                language_embedding=language_embedding, obs_embeddings=obs_embeddings, patch_embeddings=patch_embeddings)
                            # inverse diffusion step (remove noise)
                            out_action = self.noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=out_action
                            ).prev_sample
                    re_out_action = unnormalize_data(out_action)
                    self.traj_2d_list = []
                    for _ in range(10):
                        self.traj_2d_list.append(re_out_action.squeeze(0)) 
                    self.diff_flag = False
                # rgb_vis = rgb_top.permute(1, 2, 0).numpy().copy()
                # re_out_action_ori = resize_points(re_out_action.clone(), (224,224), (640,640))
                # re_out_action_ori = re_out_action_ori.squeeze(0).squeeze(0)
                # for index, point_2d in enumerate(re_out_action_ori):
                #     color = (
                #         int(255 * (index / 30)),  # 红色分量
                #         int(206 * (index / 30)),  # 绿色分量
                #         int(135 * (index / 30))   # 蓝色分量
                #     )
                #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
                #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=6, color=color, thickness=-1)
                
                # cv2.imwrite(f"traj_predict/script/visualization/image_inference_{self.rollout_step_counter}.png", rgb_vis)  
            self.rollout_step_counter += 1
            re_out_action = self.traj_2d_list[-1]
            traj_2d = torch.zeros((1, self.seq_len, 30, 2))
            traj_2d_tensor = torch.stack(self.traj_2d_list, dim=0)  # (t, c, h, w)
            traj_2d[0, :buffer_len] = traj_2d_tensor[:buffer_len]
            traj_2d = traj_2d.to(self.device)
        else:
            traj_2d = None
        # DTM 置0
        # traj_2d = torch.zeros((1, self.seq_len, 30, 2))
        # traj_2d = traj_2d.to(self.device)

        # policy inference
        with torch.no_grad():
            prediction = self.model(
                rgb_left=rgb_left_data,
                rgb_right=rgb_right_data,
                rgb_top=rgb_top_data,
                state=state_data,
                language=tokenized_text,
                action_2d = traj_2d,
                attention_mask=attention_mask
        )
        # rgb show
        # visualization image
        # p = 16
        # h_p = 14
        # w_p = 14
        # rgb_vis= rgb_right_data.reshape(shape=(rgb_right_data.shape[0], rgb_right_data.shape[1], 3, h_p, p, w_p, p)) 
        # rgb_vis = rgb_vis.permute(0, 1, 3, 5, 4, 6, 2)
        # rgb_vis = rgb_vis.reshape(shape=(rgb_vis.shape[0], rgb_vis.shape[1], h_p * w_p, (p**2) * 3))  # (b, t, n_patches, p*p*3)
        # mean = rgb_vis.mean(dim=-1, keepdim=True)
        # std = rgb_vis.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
        

        # obs_targets = prediction["obs_right_targets"]
        # obs_targets = obs_targets * std + mean
        # obs_targets = obs_targets.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
        # obs_targets = obs_targets.permute(0, 1, 6, 2, 4, 3, 5)
        # obs_targets = obs_targets.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
        # obs_targets = self.preprocessor.rgb_recovery(obs_targets)
        

        # obs_preds = prediction["obs_right_preds"]
        # obs_preds = obs_preds * std + mean
        # obs_preds = obs_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
        # obs_preds = obs_preds.permute(0, 1, 6, 2, 4, 3, 5)
        # obs_preds = obs_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
        # obs_preds = self.preprocessor.rgb_recovery(obs_preds)
        # for batch_idx in range(obs_targets.shape[0]):
        #     seq_idx = len(self.rgb_left_list)-1
        #     obs_targets_ori = obs_targets[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy()
        #     obs_targets_pred = obs_preds[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy()
        #     cv2.imwrite("visualization/obs_targets_ori.png",obs_targets_ori)
        #     cv2.imwrite("visualization/obs_targets_pred.png",obs_targets_pred)



        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
        arm_action_preds = arm_action_preds.view(-1, self.chunk_size, self.act_dim - 1)  # (t, chunk_size, act_dim - 1)
        arm_action_preds = arm_action_preds[attention_mask.flatten() > 0]

        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']  # (1, t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds.view(-1, self.chunk_size, 1)  # (t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds[attention_mask.flatten() > 0]
        gripper_action_preds = torch.nn.Sigmoid()(gripper_action_preds)
        gripper_action_preds = gripper_action_preds > 0.5
        gripper_action_preds = gripper_action_preds.int().float()
        
        if self.action_mode == 'ee_abs_pose':
            # ensembling
            if self.temporal_ensembling and self.step < self.max_publish_step:
                all_actions = torch.cat((arm_action_preds, gripper_action_preds), dim=-1)  # (all_chunk_size,act_dim,) 当前所有的预测action
                self.all_time_actions[[self.step], self.step:self.step + self.chunk_size] = all_actions[-1]
                actions_for_curr_step = self.all_time_actions[:, self.step]
                actions_populated = torch.any(actions_for_curr_step != 0, dim=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step), dtype=torch.float32)).to(self.device)
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights.unsqueeze(1)  # 增加维度以进行广播
                action_pred = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                action_pred = action_pred.detach().cpu().squeeze(0)
                if self.stats is not None:
                    #decode action_pred
                    action_pred[:7] = action_pred[:7]*self.stats['joint_std'][:7] + self.stats['joint_mean'][:7]

                    action_pred[7] = action_pred[7] > 0.5
                    action_pred[7] = action_pred[7].int().float()
                self.step += 1 
            else:
                # Use the last action
                arm_action_pred = arm_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
                gripper_action_pred = gripper_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, 1)
                action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (act_dim,)
                action_pred = action_pred.detach().cpu().squeeze(0)
                if self.stats is not None:
                    #decode action_pred
                    action_pred[:7] = action_pred[:7]*self.stats['joint_std'][:7] + self.stats['joint_mean'][:7]
        else: # 'ee_rel_pose'
            # ensembling
            if self.temporal_ensembling and self.step < self.max_publish_step:
                all_actions = torch.cat((arm_action_preds, gripper_action_preds), dim=-1)  # (all_chunk_size,act_dim,) 当前所有的预测action
                self.all_time_actions[[self.step], self.step:self.step + self.chunk_size] = all_actions[-1]
                actions_for_curr_step = self.all_time_actions[:, self.step]
                actions_populated = torch.any(actions_for_curr_step != 0, dim=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step), dtype=torch.float32)).to(self.device)
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights.unsqueeze(1)  # 增加维度以进行广播
                action_pred = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                action_pred = action_pred.detach().cpu().squeeze(0)
                if self.stats is not None:
                    #decode action_pred
                    action_pred[:7] = action_pred[:7]*self.stats['delta_joint_std'][:7] + self.stats['delta_joint_mean'][:7]
                action_pred[:7] = action_pred[:7] + arm_state
                action_pred[7] = action_pred[7] > 0.5
                action_pred[7] = action_pred[7].int().float()
                self.step += 1 
            else:
                # Use the last action
                arm_action_pred = arm_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
                gripper_action_pred = gripper_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, 1)
                action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (act_dim,)
                action_pred = action_pred.detach().cpu().squeeze(0)
                if self.stats is not None:
                    action_pred[:7] = action_pred[:7]*self.stats['delta_joint_std'][:7] + self.stats['delta_joint_mean'][:7]
                action_pred[:7] = action_pred[:7] + arm_state
        # visualize 2d traj
        rgb_vis = rgb_top.permute(1, 2, 0).numpy().copy()
        if self.use_2d_traj:
            point_2d_pred = prediction['traj_2d_preds'][0][-1] * 640
            re_out_action_ori = resize_points(re_out_action.clone(), (224,224), (640,640))
            re_out_action_ori = re_out_action_ori.squeeze(0).squeeze(0)
            for index, point_2d in enumerate(re_out_action_ori):
                color = (
                    int(255 * (index / 30)),  # 红色分量
                    int(206 * (index / 30)),  # 绿色分量
                    int(135 * (index / 30))   # 蓝色分量
                )
                cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
                cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=6, color=color, thickness=-1)
            # 2d traj 5 points prediction
            for point_2d in point_2d_pred :
                cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=3, color=(255, 0, 0), thickness=-1)
            # 检测按键输入
            key = cv2.waitKey(1) & 0xFF  # 等待键盘输入

            if key == ord('r'):  # 如果按下 'K'
                self.diff_flag = True
                print("Key 'r' pressed, diff_flag set to:", self.diff_flag)
        cv2.imshow("image_inference", rgb_vis)
        cv2.waitKey(1)

        return action_pred
def main_DTP():
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = json.load(open('Real_Robot_configs_inference.json'))
    DTP_Eva = DTP_Evaluation(cfg)
    print('DTP model loaded')
    

    DTP_Eva.reset()

    # fake
    # fake_left_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # _, fake_left_image = cv2.imencode('.jpg', fake_left_image)
    # fake_right_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # _, fake_right_image = cv2.imencode('.jpg', fake_right_image)
    # fake_top_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # _, fake_top_image = cv2.imencode('.jpg', fake_top_image)

    frame_num = 0
    while True:
        image_left_path = f'traj_predict/script/visualization/rgb/camera_left_{frame_num}.jpg'
        fake_left_image = cv2.imread(image_left_path)
        _, fake_left_image = cv2.imencode('.jpg', fake_left_image)
        image_right_path = f'traj_predict/script/visualization/rgb/camera_right_{frame_num}.jpg'
        fake_right_image = cv2.imread(image_right_path)
        _, fake_right_image = cv2.imencode('.jpg', fake_right_image)
        image_top_path = f'traj_predict/script/visualization/rgb/camera_top_{frame_num}.jpg'
        fake_top_image = cv2.imread(image_top_path)
        _, fake_top_image = cv2.imencode('.jpg', fake_top_image)



        fake_obs = {
            'images': {
                'left': fake_left_image,
                'right':fake_right_image,  
                'top':fake_top_image,
            },
            'qpos': np.array([-0.296549528837204, 0.24149464070796967, 0.2143513560295105, -1.4751654863357544, -0.08158532530069351, 1.718814730644226, -0.3496069312095642, 0.0]),
            
            # 'pose': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'pose': np.array([0.5720672011375427, -0.04308487847447395, 0.5743494629859924, 3.0456106722554175, 0.06872873345163044, -0.08510535636997513]),
        }

        action = DTP_Eva.inference(fake_obs)
        frame_num += 1
        if frame_num > 110:
            frame_num = 0
        print(action)



    
if __name__ == "__main__":
    # main()
    main_DTP()
