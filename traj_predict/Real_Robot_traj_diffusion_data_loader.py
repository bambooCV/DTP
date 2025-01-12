import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
from torch.utils.data import DataLoader
import random
import cv2 
ORIGINAL_STATIC_RES_W = 640
ORIGINAL_STATIC_RES_H = 480

def contains_words(inst, include_words=[], exclude_words=[]):
    for word in include_words:
        if word not in inst:
            return False
    for word in exclude_words:
        if word in inst:
            return False
    return True
def add_noise(sequence, noise_level=2.0):
    noise = torch.randn(sequence.shape) * noise_level
    noisy_sequence = sequence + noise
    return noisy_sequence
def resample_sequence(sequence, target_length):
    """
    使用插值将 sequence 重新采样到 target_length，并将结果四舍五入为整数
    :param sequence: 原始序列，形状为 [N, 2]
    :param target_length: 目标序列长度
    :return: 重新采样后的序列，形状为 [target_length, 2]
    """
    # 确保 sequence 是 float 类型
    sequence = sequence.float()
    sequence = sequence.unsqueeze(0).permute(0, 2, 1)  # 调整形状为 (1, 2, N)
    resampled_sequence = F.interpolate(sequence, size=target_length, mode='linear', align_corners=True)
    resampled_sequence = resampled_sequence.permute(0, 2, 1).squeeze(0)  # 调整回原始形状 (target_length, 2)
    resampled_sequence = add_noise(resampled_sequence, noise_level=0.75)
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
    
def visulization_image(rgb_static,actions,inst):
  
    rgb_static_rgb = cv2.cvtColor(rgb_static.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    # for point_2d in future_2d_actions[:,:2]:
    #     cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 255, 255), thickness=-1)
    for point_2d in actions:
        cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
    # 把inst的文字放在图片左下角 放在左下角！

    cv2.putText(rgb_static_rgb, inst, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
    cv2.imshow('Processed RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式


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
                if isinstance(self.batch[key], torch.Tensor):
                    self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time

class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, sequence_length, chunk_size, action_dim, start_ratio, end_ratio):
        super(LMDBDataset).__init__()
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.dummy_rgb_camera_top_padding = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES_W, ORIGINAL_STATIC_RES_W, dtype=torch.uint8)
        self.dummy_top_actions = torch.zeros(sequence_length, chunk_size, action_dim)
        self.dummy_mask = torch.zeros(sequence_length)
        self.lmdb_dir = lmdb_dir

        env = lmdb.open(lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()

        idx = idx + self.start_step

        rgb_camera_top = self.dummy_rgb_camera_top_padding.clone()
        top_actions = self.dummy_top_actions.clone()
        mask = self.dummy_mask.clone()

        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        inst_token = loads(self.txn.get(f'inst_token_{cur_episode}'.encode()))
        inst = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
        inst_emb = loads(self.txn.get(f'inst_emb_{cur_episode}'.encode()))
        for i in range(self.sequence_length):
            new_idx = idx + i
            if loads(self.txn.get(f'cur_episode_{new_idx}'.encode())) == cur_episode:
                mask[i] = 1
                get_rgb_camera_top = decode_jpeg(loads(self.txn.get(f'rgb_camera_top_{new_idx}'.encode())))
                get_rgb_camera_top = F.interpolate(get_rgb_camera_top.unsqueeze(0), size=(ORIGINAL_STATIC_RES_H, ORIGINAL_STATIC_RES_W), mode='bilinear', align_corners=False).squeeze(0)
                rgb_camera_top[i] = F.pad(
                                            get_rgb_camera_top,
                                            pad=(0, 0, 80, 80),  # 在高度方向填充
                                            mode='replicate'  # 使用边缘像素填充
                                        )
                traj_2d_top = loads(self.txn.get(f'traj_2d_top_{new_idx}'.encode()))
                if len(traj_2d_top) < self.chunk_size/3:
                    mask[i] = 0  
                else: 
                    traj_2d_top_trans = traj_2d_top * torch.tensor([0.5, 0.667])  + torch.tensor([0, 80]) # 坐标转换
                    top_actions[i,:,:] = resample_sequence_adapter(traj_2d_top_trans, self.chunk_size) # 坐标下采样
                    # vis
                    # rgb_vis = rgb_camera_top[i].permute(1, 2, 0).numpy().copy()
                    # for (u, v) in top_actions[i]:
                    #     cv2.circle(rgb_vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1) 

                    # cv2.imwrite("traj_predict/script/visualization/rgb_vis.png", rgb_vis)  
                    # print('test')

                
        return {
            'rgb_camera_top': rgb_camera_top,
            'inst':inst,
            'inst_token': inst_token,
            'inst_emb': inst_emb,
            'top_actions': top_actions,
            'mask': mask,
        }

    def __len__(self):
        return self.end_step - self.start_step

if __name__ == '__main__':
    from traj_func import Real_Robot_2D_PreProcess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 96
    num_workers = 1
    preprocessor = Real_Robot_2D_PreProcess(
        rgb_static_pad = 10, # 去除位置敏感性
        rgb_shape = [224,224], 
        rgb_mean = [0.485, 0.456, 0.406],
        rgb_std =  [0.229, 0.224, 0.225],
        device = device
    )

    train_dataset = LMDBDataset(
        lmdb_dir = "/bamboo_dir/pick_bread_plate/success_episodes_lmdb/",
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过65
        action_dim = 2, # x,y,gripper_state
        start_ratio = 0,
        end_ratio = 0.28, 
    )
    val_dataset = LMDBDataset(
        lmdb_dir = "/bamboo_dir/pick_bread_plate/success_episodes_lmdb/",
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过65
        action_dim = 2,
        start_ratio = 0.28,
        end_ratio = 0.3, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, # to be flattened in prefetcher  
        num_workers=num_workers,
        pin_memory=True, # Accelerate data reading  
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
    ) 
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, # to be flattened in prefetcher  
        num_workers=num_workers,
        pin_memory=True, # Accelerate data reading
        shuffle=False,
        prefetch_factor=2,
        persistent_workers=True,
    ) 
    train_prefetcher = DataPrefetcher(train_loader, device)
    test_prefetcher = DataPrefetcher(val_loader, device)

    from tqdm import tqdm
    for epoch in range(1):
        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}", leave=False) as pbar:
            batch, load_time = train_prefetcher.next()
            while batch is not None:
                batch, load_time = train_prefetcher.next()
                image = batch['rgb_camera_top']
                naction = batch['top_actions']
                rgb_top_norm,naction_transformed = preprocessor.rgb_process(batch['rgb_camera_top'],batch['top_actions'],train=True)
                # visualization croped image
                # Convert tensor to NumPy array for visualization
                import cv2
                for batch_idx in range(image.shape[0]):
                    for seq_idx in range(image.shape[1]):
                        rgb_camera_top = image[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy().copy()
                        for point_2d in naction[batch_idx,seq_idx,:,:]:
                            cv2.circle(rgb_camera_top, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                        cv2.putText(rgb_camera_top, batch["inst"][batch_idx], (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
                        cv2.imwrite("traj_predict/script/visualization/rgb_camera_top_groundtruth.png", rgb_camera_top)  

                        rgb_top_reshape = preprocessor.rgb_recovery(rgb_top_norm)
                        rgb_top_np = rgb_top_reshape[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy().copy()

                        for point_2d in naction_transformed[batch_idx,seq_idx,:,:]:
                            cv2.circle(rgb_top_np, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                        cv2.imwrite("traj_predict/script/visualization/rgb_camera_top_trans.png", rgb_top_np)  
                        cv2.waitKey(10)
                pbar.update(1) 

