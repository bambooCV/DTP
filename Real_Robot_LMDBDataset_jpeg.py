import json
import os
from time import time
import lmdb
from pickle import loads
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PreProcess import Real_Robot_PreProcess
import torchvision.transforms as transforms
import cv2
from Real_Robot.read_franka_h5 import ReadH5Files
ORIGINAL_STATIC_RES_W = 640
ORIGINAL_STATIC_RES_H = 480

def add_noise(sequence, noise_level=2.0):
    noise = torch.randn(sequence.shape) * noise_level
    noisy_sequence = sequence + noise
    return noisy_sequence
def resample_sequence(sequence, target_length):
    """
    使用插值将 sequence 重新采样到 target_length,并将结果四舍五入为整数
    :param sequence: 原始序列，形状为 [N, 2]
    :param target_length: 目标序列长度
    :return: 重新采样后的序列，形状为 [target_length, 2]
    """
    # 确保 sequence 是 float 类型
    sequence = sequence.float()
    sequence = sequence.unsqueeze(0).permute(0, 2, 1)  # 调整形状为 (1, 2, N)
    resampled_sequence = F.interpolate(sequence, size=target_length, mode='linear', align_corners=True)
    resampled_sequence = resampled_sequence.permute(0, 2, 1).squeeze(0)  # 调整回原始形状 (target_length, 2)
    # resampled_sequence = add_noise(resampled_sequence, noise_level=0.75)
    # 将结果四舍五入为整数
    resampled_sequence = torch.round(resampled_sequence).int()
    
    return resampled_sequence


def resample_sequence_adapter(sequence, target_length):
    """
    使用自适应插值将 sequence 重新采样到 target_length，并将结果四舍五入为整数。
    稀疏区域分配更多采样点，密集区域分配更少采样点。
    
    :param sequence: 原始序列，形状为 [N, 2]
    :param target_length: 目标序列长度
    :return: 重新采样后的序列，形状为 [target_length, 2]
    """
    # Step 1: 确保 sequence 是 float 类型
    sequence = sequence.float()

    # Step 2: 计算点间距离
    distances = torch.norm(sequence[1:] - sequence[:-1], dim=1)  # 相邻点距离
    distances = torch.cat((torch.tensor([0.0]), distances))  # 插入首点距离为0

    # Step 3: 计算累积分布函数 (CDF)
    cumulative_distances = torch.cumsum(distances.clone(), dim=0)  # 累计距离
    cumulative_distances = cumulative_distances.clone()  # 再次克隆确保独立内存
    cumulative_distances = cumulative_distances / cumulative_distances[-1]  # 归一化到 [0, 1]

    # Step 4: 生成目标网格点
    target_positions = torch.linspace(0, 1, target_length)  # 目标均匀分布的点

    # Step 5: 使用插值计算采样点
    indices = torch.searchsorted(cumulative_distances, target_positions, right=True)
    indices = torch.clamp(indices, 1, len(sequence) - 1)  # 防止索引越界

    # 获取相邻点的索引及插值权重
    left_indices = indices - 1
    right_indices = indices
    left_weights = (cumulative_distances[right_indices] - target_positions) / (
        cumulative_distances[right_indices] - cumulative_distances[left_indices]
    )
    right_weights = 1 - left_weights

    # 插值计算新的采样点
    resampled_sequence = (
        left_weights.unsqueeze(1) * sequence[left_indices] +
        right_weights.unsqueeze(1) * sequence[right_indices]
    )

    # Step 6: 添加噪声（如有需要）
    resampled_sequence = add_noise(resampled_sequence, noise_level=0.75)

    # Step 7: 将结果四舍五入为整数
    resampled_sequence = torch.round(resampled_sequence).int()
    
    return resampled_sequence
class DataPrefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        with torch.cuda.stream(self.stream):
            for key in self.batch:
                self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time

def get_files(dataset_dir, robot_infor):
    read_h5files = ReadH5Files(robot_infor)
    dataset_dir = os.path.join(dataset_dir, 'success_episodes')
    files = []
    all_action_data = []

    for trajectory_id in sorted(os.listdir(dataset_dir)):
        trajectory_dir = os.path.join(dataset_dir, trajectory_id)
        file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
        try:
            _, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(file_path, camera_frame=2)
            files.append(file_path)
            action = control_dict['puppet']['joint_position'][:]
            all_action_data.append(torch.from_numpy(action))
        except Exception as e:
            print(e)
    return files,all_action_data


class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, sequence_length, chunk_size, action_mode, action_dim, start_ratio, end_ratio):
        super(LMDBDataset).__init__()
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.dummy_rgb_camera_left = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES_H, ORIGINAL_STATIC_RES_W, dtype=torch.uint8)
        self.dummy_rgb_camera_right = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES_H, ORIGINAL_STATIC_RES_W, dtype=torch.uint8)
        self.dummy_rgb_camera_top = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES_H, ORIGINAL_STATIC_RES_W, dtype=torch.uint8)
        self.dummy_rgb_camera_left_padding = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES_W, ORIGINAL_STATIC_RES_W, dtype=torch.uint8)
        self.dummy_rgb_camera_right_padding = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES_W, ORIGINAL_STATIC_RES_W, dtype=torch.uint8)
        self.dummy_rgb_camera_top_padding = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES_W, ORIGINAL_STATIC_RES_W, dtype=torch.uint8)

        self.dummy_arm_state = torch.zeros(sequence_length, 7)
        self.dummy_gripper_state =  torch.zeros(sequence_length, 2)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, action_dim)
        self.dummy_actions_2d = torch.zeros(sequence_length,30, 2)
        self.dummy_traj_2d_preds = torch.zeros(sequence_length,100, 2) # 最多100个2d点
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.lmdb_dir = lmdb_dir
        env = lmdb.open(lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length - chunk_size
            self.norm_stats = loads(txn.get(b'stats'))
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()

        idx = idx + self.start_step
        rgb_camera_left = self.dummy_rgb_camera_left_padding.clone()
        rgb_camera_right = self.dummy_rgb_camera_right_padding.clone()
        rgb_camera_top = self.dummy_rgb_camera_top_padding.clone()

        arm_state = self.dummy_arm_state.clone()
        gripper_state = self.dummy_gripper_state.clone()
        actions = self.dummy_actions.clone()
        mask = self.dummy_mask.clone()
        actions_2d = self.dummy_actions_2d.clone()
        traj_2d_preds =self.dummy_traj_2d_preds.clone()

        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        inst = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
        inst_token = loads(self.txn.get(f'inst_token_{cur_episode}'.encode()))
        for i in range(self.sequence_length):
            if loads(self.txn.get(f'cur_episode_{idx+i}'.encode())) == cur_episode:
                get_rgb_camera_left = decode_jpeg(loads(self.txn.get(f'rgb_camera_left_{idx+i}'.encode())))
                rgb_camera_left[i] = F.pad(
                                        get_rgb_camera_left,
                                        pad=(0, 0, 80, 80),  # 在高度方向填充
                                        mode='replicate'  # 使用边缘像素填充
                                    )
                get_rgb_camera_right = decode_jpeg(loads(self.txn.get(f'rgb_camera_right_{idx+i}'.encode())))
                rgb_camera_right[i] = F.pad(
                                        get_rgb_camera_right,
                                        pad=(0, 0, 80, 80),  # 在高度方向填充
                                        mode='replicate'  # 使用边缘像素填充
                                    )
                get_rgb_camera_top = decode_jpeg(loads(self.txn.get(f'rgb_camera_top_{idx+i}'.encode())))
                get_rgb_camera_top = F.interpolate(get_rgb_camera_top.unsqueeze(0), size=(ORIGINAL_STATIC_RES_H, ORIGINAL_STATIC_RES_W), mode='bilinear', align_corners=False).squeeze(0)
                rgb_camera_top[i] = F.pad(
                                            get_rgb_camera_top,
                                            pad=(0, 0, 80, 80),  # 在高度方向填充
                                            mode='replicate'  # 使用边缘像素填充
                                        )
                # visualization
                # rgb_camera_left_padding = rgb_camera_left[i].permute(1, 2, 0).cpu().numpy()
                # rgb_camera_right_padding = rgb_camera_right[i].permute(1, 2, 0).cpu().numpy()
                # rgb_camera_top_padding = rgb_camera_top[i].permute(1, 2, 0).cpu().numpy()

                # cv2.imwrite("visualization/rgb_camera_left_padding.png",rgb_camera_left_padding)
                # cv2.imwrite("visualization/rgb_camera_right_padding.png",rgb_camera_right_padding)
                # cv2.imwrite("visualization/rgb_camera_top_padding.png",rgb_camera_top_padding)
                # end
                robot_obs = loads(self.txn.get(f'end_effector_{idx+i}'.encode()))
                robot_obs_gripper = loads(self.txn.get(f'joint_position_{idx+i}'.encode()))[-1]
                arm_state[i, :7] = loads(self.txn.get(f'joint_position_{idx+i}'.encode()))[:7] 
                gripper_state[i, 0 if robot_obs_gripper < 0.2 else 1] = 1
                for j in range(self.chunk_size):
                    if loads(self.txn.get(f'cur_episode_{idx+i+j}'.encode())) == cur_episode:
                        mask[i, j] = 1
                        if self.action_mode == 'ee_rel_pose':
                            actions[i, j] = loads(self.txn.get(f'delta_joint_position_{idx+i+j}'.encode()))
                            if self.norm_stats is not None:
                                # action processing
                                actions[i, j,:7] = (actions[i, j, :7] - self.norm_stats["delta_joint_mean"][:7]) / self.norm_stats["delta_joint_std"][:7]
                            actions[i, j, -1] = 0 if actions[i, j, -1] < 0.2 else 1
                        else:
                            actions[i, j] = loads(self.txn.get(f'joint_position_{idx+i+j}'.encode()))
                            if self.norm_stats is not None:
                                # action processing
                                actions[i, j,:7] = (actions[i, j, :7] - self.norm_stats["joint_mean"][:7]) / self.norm_stats["joint_std"][:7]

                            actions[i, j, -1] = 0 if actions[i, j, -1] < 0.2 else 1

                # traj_2d data
                traj_2d_preds_len =len(loads(self.txn.get(f'traj_2d_top_{idx+i}'.encode())))
                traj_2d_preds[i,:min(100,traj_2d_preds_len)] = loads(self.txn.get(f'traj_2d_top_{idx+i}'.encode()))[:min(100,traj_2d_preds_len)]
                if traj_2d_preds_len < 100:
                    traj_2d_preds[i,traj_2d_preds_len:] = traj_2d_preds[i,traj_2d_preds_len-1] # 补齐长度
                traj_2d_preds[i] = traj_2d_preds[i] * torch.tensor([0.5, 0.667])  + torch.tensor([0, 80]) # 坐标转换
                future_2d_actions = loads(self.txn.get(f'traj_2d_top_init_{cur_episode}'.encode())) # 只用当前episode的2d action 为了和inference保持一致 且快速验证
                future_2d_actions = future_2d_actions * torch.tensor([0.5, 0.667])  + torch.tensor([0, 80]) # 坐标转换
                actions_2d[i] = resample_sequence_adapter(future_2d_actions, 30)
                # import cv2
                # image =  rgb_camera_top[i] 
                # action = actions_2d[i] 
                # rgb_vis = image.permute(1, 2, 0).cpu().numpy().copy()
                # for index, point_2d in enumerate(action):
                #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=8, color=(255, 255, 255), thickness=-1)
                #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=6, color=(225,206,135), thickness=-1)
                # cv2.imwrite("traj_predict/script/visualization/tmp.png", rgb_vis)  
                # print('fsc_test')

        return {

            'rgb_camera_left': rgb_camera_left,
            'rgb_camera_right': rgb_camera_right,
            'rgb_camera_top': rgb_camera_top,
            # 'inst': inst,
            'inst_token': inst_token,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'actions': actions,
            'mask': mask,
            'actions_2d': actions_2d,
            'traj_2d_preds': traj_2d_preds,
        }

    def __len__(self):
        return self.end_step - self.start_step
def get_norm_stats(dataset_dir):
    # normlization
    robot_infor = {'camera_names': ['camera_left', 'camera_right'],
        'camera_sensors': ['rgb_images'],
        'arms': ['puppet'],
        'controls': ['joint_position',  'end_effector']}
    
    files,all_action_data = get_files(dataset_dir, robot_infor)
    # normalize action data
    all_action_data = torch.cat(all_action_data, dim=0)
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    eps = 0.0001
    stats = {"action_mean": action_mean, "action_std": action_std,
            "action_min": action_min - eps, "action_max": action_max + eps}
    return stats
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     # Preparation
    cfg = json.load(open('Real_Robot_configs_debug.json'))
    # 获得stats
    env = lmdb.open(cfg['LMDB_path'], readonly=True, create=False, lock=False)
    with env.begin() as txn:
        stats = loads(txn.get(b'stats'))
    env.close()
    
    train_dataset = LMDBDataset(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['chunk_size'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = 0,
        end_ratio = 0.95, 
    )
    test_dataset = LMDBDataset(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['chunk_size'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = 0.95,
        end_ratio = 1, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        # shuffle=True,
        shuffle=False,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        shuffle=False,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    train_prefetcher = DataPrefetcher(train_loader, device)
    test_prefetcher = DataPrefetcher(test_loader, device)
    for epoch in range(10):
        batch, load_time = train_prefetcher.next()
        delta_actions = []
        while batch is not None:
            
            print(batch['actions'])
            # visualization
            # rgb_image_left = batch['rgb_camera_left'][0][0].permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
            # rgb_image_right = batch['rgb_camera_right'][0][0].permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
            # rgb_image_top = batch['rgb_camera_top'][0][0].permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
            # cv2.imwrite('visualization/rgb_camera_left.png', rgb_image_left)
            # cv2.imwrite('visualization/rgb_image_right.png', rgb_image_right)
            # cv2.imwrite('visualization/rgb_image_top.png', rgb_image_top)
            delta_actions.append(batch['actions'])
            batch, load_time = train_prefetcher.next_without_none()

        all_delta_actions = torch.cat(delta_actions)
        min_value = torch.min(all_delta_actions)
        max_value = torch.max(all_delta_actions)
        print("Minimum value:", min_value.item())
        print("Maximum value:", max_value.item())