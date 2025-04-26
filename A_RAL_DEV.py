import warnings
import sys
sys.path.append('..')
# sys.path.append('/home/ps/Dev/inrocs/')
from einops import rearrange

import torch
import cv2
import json

import numpy as np

from models.ral_real_robot_gr1 import GR1 
# from robot_env.franka_env import robot_env
import models.vision_transformer as vits
import torch.nn.functional as F
from PreProcess import Real_Robot_PreProcess
# from leader.agent import LeaderAgent, BiLeaderAgent
# print('aaa')
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
import clip
preparing = True


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
            use_multi_rgb=True,
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

        # ensembling
        self.temporal_ensembling = True
        self.max_publish_step = 10000
        self.step = 0
        self.all_time_actions = torch.zeros([self.max_publish_step, self.max_publish_step + self.chunk_size, self.state_dim]).to(self.device)

    def reset(self):
        """Reset function."""
        self.rgb_left_list = []
        self.rgb_wrist_list = []
        self.rgb_top_list = []
        self.state_list = []
        self.rollout_step_counter = 0
        
        self.step = 0
        self.all_time_actions = torch.zeros([self.max_publish_step, self.max_publish_step + self.chunk_size, self.state_dim]).to(self.device)

    def infer(self,obs,task_name='open the upper drawer'):
        # Language
        tokenized_text = self.tokenizer(text)

        # get images
        all_cam_images =[]
        for cam_name in ['left', 'wrist', 'front']:

            cam_img = obs['images'][cam_name]
            cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)
     
            padding_value = (640 - 480) // 2  # 假设宽度大于高度
            cam_img = cv2.copyMakeBorder(
                cam_img,
                top=padding_value, bottom=padding_value, left=0, right=0,  # 只在高度方向填充
                borderType=cv2.BORDER_REPLICATE  # 使用边缘填充
            )

            # cv2.imwrite(f"/home/ps/Dev/bamboo/inrocs_online_inference/run/visualization/cam_img_padding_{cam_name}.png",cam_img)
            all_cam_images.append(cam_img)
        
        # RGB
        rgb_left = rearrange(torch.from_numpy(all_cam_images[0]), 'h w c -> c h w')
        rgb_wrist = rearrange(torch.from_numpy(all_cam_images[1]), 'h w c -> c h w')
        rgb_top = rearrange(torch.from_numpy(all_cam_images[2]), 'h w c -> c h w')
        self.rgb_left_list.append(rgb_left)
        self.rgb_wrist_list.append(rgb_wrist)
        self.rgb_top_list.append(rgb_top)

        # State
        arm_state = obs['arm_joints']['single'][:7]
        # arm_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        gripper_state = 0 if obs['arm_joints']['single'][-1] < 0.2 else 1
        state = torch.from_numpy(np.hstack([arm_state, gripper_state]))
        self.state_list.append(state)

        # Buffer
        buffer_len = len(self.rgb_left_list)
        if buffer_len > self.seq_len:
            self.rgb_left_list.pop(0)
            self.rgb_wrist_list.pop(0)
            self.rgb_top_list.pop(0)
            self.state_list.pop(0)
            assert len(self.rgb_left_list) == self.seq_len
            assert len(self.rgb_wrist_list) == self.seq_len
            assert len(self.rgb_top_list) == self.seq_len
            assert len(self.state_list) == self.seq_len
            buffer_len = len(self.rgb_left_list)
        
        # left RGB
        c, h, w = rgb_left.shape
        rgb_left_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_left_tensor = torch.stack(self.rgb_left_list, dim=0)  # (t, c, h, w)
        rgb_left_data[0, :buffer_len] = rgb_left_tensor

        # wrist RGB
        c, h, w = rgb_wrist.shape
        rgb_wrist_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_wrist_tensor = torch.stack(self.rgb_wrist_list, dim=0)  # (t, c, h, w)
        rgb_wrist_data[0, :buffer_len] = rgb_wrist_tensor

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
        rgb_wrist_data = rgb_wrist_data.to(self.device)
        rgb_top_data = rgb_top_data.to(self.device)
        arm_state_data = arm_state_data.to(self.device)
        gripper_state_data = gripper_state_data.to(self.device)
        state_data = {'arm': arm_state_data, 'gripper': gripper_state_data}
        attention_mask = attention_mask.to(self.device)
        rgb_left_data, rgb_wrist_data, rgb_top_data = self.preprocessor.rgb_process(rgb_left_data, rgb_wrist_data, rgb_top_data, train=False)
        
        with torch.no_grad():
            prediction = self.model(
                rgb_left=rgb_left_data,
                rgb_wrist=rgb_wrist_data,
                rgb_top=rgb_top_data,
                state=state_data,
                language=tokenized_text,
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
            # cv2.imwrite("/home/ps/Dev/bamboo/inrocs_online_inference/run/visualization/obs_targets_ori.png",obs_targets_ori)
            # cv2.imwrite("/home/ps/Dev/bamboo/inrocs_online_inference/run/visualization/obs_targets_pred.png",obs_targets_pred)



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
  
        return action_pred
def main_DTP():
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = json.load(open('A_RAL_DEV.json'))
    DTP_Eva = DTP_Evaluation(cfg)
    print('DTP model loaded')
    

    DTP_Eva.reset()
    # frame_num = 1
    # image_left_path = f'traj_predict/script/visualization/rgb/camera_left_{frame_num}.jpg'
    # fake_left_image = cv2.imread(image_left_path)
    # _, fake_left_image = cv2.imencode('.jpg', fake_left_image)
    # image_right_path = f'traj_predict/script/visualization/rgb/camera_right_{frame_num}.jpg'
    # fake_right_image = cv2.imread(image_right_path)
    # _, fake_right_image = cv2.imencode('.jpg', fake_right_image)
    # image_top_path = f'traj_predict/script/visualization/rgb/camera_top_{frame_num}.jpg'
    # fake_top_image = cv2.imread(image_top_path)
    # _, fake_top_image = cv2.imencode('.jpg', fake_top_image)

    # fake
    fake_left_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    _, fake_left_image = cv2.imencode('.jpg', fake_left_image)
    fake_wrist_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    _, fake_wrist_image = cv2.imencode('.jpg', fake_wrist_image)
    fake_top_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    _, fake_top_image = cv2.imencode('.jpg', fake_top_image)

    fake_obs = {
        'images': {
            'left': fake_left_image,
            'wrist':fake_wrist_image,  
            'top':fake_top_image,
        },
        'qpos': np.array([-0.04249195754528046, 0.06590165197849274, -0.016230255365371704, -1.6042454242706299, -0.09310498088598251, 1.5987823009490967, 0.02900964766740799, 0.0]),
        # 'pose': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'pose': np.array([0.5720672011375427, -0.04308487847447395, 0.5743494629859924, 3.0456106722554175, 0.06872873345163044, -0.08510535636997513]),
    }
    # action_mean = torch.tensor([ 0.0224,  0.2196, -0.0879, -1.6439, -0.0494,  1.8743, -0.0359,  0.1230])
    # action_std = torch.tensor([0.1829, 0.2051, 0.1402, 0.1658, 0.0625, 0.2278, 0.2456, 0.1852])
    # stats = {"action_mean": action_mean, "action_std": action_std}
    while True:
        action = DTP_Eva.inference(fake_obs)


        print(action)



    
if __name__ == "__main__":
    # main()
    main_DTP()







# import time
# import numpy as np
# from xrocs.common.data_type import Joints
# from xrocs.core.config_loader import ConfigLoader
# from xrocs.utils.logger.logger_loader import logger
# from xrocs.core.station_loader import StationLoader

# # 导入模型class
# from delpoyment.inference_frank_neil_diffusion_test import DiffusionPolicyInference


# class JointInference:
#     def __init__(self, config_path = None, model_path = None):
#         if config_path == None:
#             config_path = "/home/eai/Documents/configuration.toml"
#         cfg_loader = ConfigLoader(config_path)
#         self.cfg_dict = cfg_loader.get_config()
#         station_loader = StationLoader(self.cfg_dict)
#         self.robot_station = station_loader.get_station_handle()
#         self.robot_station.connect()
#         self.infer_model = DiffusionPolicyInference(model_path)

#     def prepare(self):
#         for name, _robot in self.robot_station.get_robot_handle().items():
#             home = Joints(self.cfg_dict['robot']['arm']['home'][name],
#                           num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
#             _robot.reach_target_joint(home)
#         for gripper in self.robot_station.get_gripper_handle().values():
#             gripper.open()
#         time.sleep(2)
#         logger.success('Resetting to home success!')

#     def inference(self, data_dir: str, task_name: str):
#         obs = self.robot_station.get_obs()
#         while True:
#             action_pred: np.ndarray = self.infer_model.infer(obs)
#             action_pred = action_pred[0].cpu().numpy()

#             robot_targets = self.robot_station.decompose_action(action_pred)
#             obs = self.robot_station.step(robot_targets)

# if __name__ == '__main__':
#     config_path = '~/Documents/configuration.toml'
#     data = JointInference('/home/eai/ckpt/pretrained_model')
#     data.prepare()
#     data.inference('/home/eai/data/example_dir',"example_task")