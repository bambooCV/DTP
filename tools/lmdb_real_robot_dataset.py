import sys
sys.path.insert(0, '/bamboo.fan/EmbodiedAI/calvin')
import os
import io
import argparse
import lmdb
from pickle import dumps, loads
import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.io import encode_jpeg
import clip
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import cv2
import pickle
from collections import defaultdict
import h5py
from scipy.spatial.transform import Rotation 
from pathlib import Path
class ReadH5Files():
    def __init__(self, robot_infor):
        self.camera_names = robot_infor['camera_names']
        self.camera_sensors = robot_infor['camera_sensors']

        self.arms = robot_infor['arms']
        self.robot_infor = robot_infor['controls']

        # 'joint_velocity_left', 'joint_velocity_right',
        # 'joint_effort_left', 'joint_effort_right',
        pass

    def decoder_image(self, camera_rgb_images, camera_depth_images=None):
        if type(camera_rgb_images[0]) is np.uint8:
            # print(f"0 camera_rgb_images size:{camera_rgb_images.shape}")
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            # print(f"1 rgb size:{rgb.shape}")
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                # print(f"2 rgb size:{rgb.shape}")
                if camera_depth_images is not None:
                    depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                    depth_images.append(depth)
                else:
                    depth_images = np.asarray([])
                rgb_images.append(rgb)
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images)
            return rgb_images, depth_images

    def execute(self, file_path, camera_frame=None, control_frame=None, use_depth_image=False):
        image_dict = defaultdict(dict)
        control_dict = defaultdict(dict)
        base_dict = defaultdict(dict)
        with h5py.File(file_path, 'r') as root:
            is_sim = root.attrs['sim']
            is_compress = True
            # select camera frame id
            for cam_name in self.camera_names:

                if camera_frame is not None:
                    if use_depth_image:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                        image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                        image_dict[self.camera_sensors[1]][cam_name] = decode_depth
                    else:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                            camera_depth_images=None)
                        # decode_rgb如果None 说明没压缩，直接获取图像结果
                        if decode_rgb is None:
                            decode_rgb = root['observations'][self.camera_sensors[0]][cam_name][camera_frame]
                            if decode_rgb.size == 640*480*3:
                                decode_rgb = decode_rgb.reshape(480,640,3)
                            elif decode_rgb.size == 1280*720*3:
                                decode_rgb = decode_rgb.reshape(720,1280,3)
                        image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                else:
                    if use_depth_image:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][()],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][()])
                        image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                        image_dict[self.camera_sensors[1]][cam_name] = decode_depth
                    else:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                            camera_depth_images=None)
                        # decode_rgb如果None 说明没压缩，直接获取图像结果
                        if decode_rgb is None:
                            decode_rgb = root['observations'][self.camera_sensors[0]][cam_name][camera_frame]
                            if decode_rgb.size == 640*480*3:
                                decode_rgb = decode_rgb.reshape(480,640,3)
                            elif decode_rgb.size == 1280*720*3:
                                decode_rgb = decode_rgb.reshape(720,1280,3)
                        image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                if decode_rgb.shape != (480,640,3) and decode_rgb.shape != (720,1280,3):
                    print(f"ERROR decode_rgb size: {decode_rgb.shape}, file_path:{file_path}")

            for arm_name in self.arms:
                for control in self.robot_infor:
                    if control_frame:
                        control_dict[arm_name][control] = root[arm_name][control][control_frame]
                    else:
                        control_dict[arm_name][control] = root[arm_name][control][()]
            # print('infor_dict:',infor_dict)
        # gc.collect()
        return image_dict, control_dict, base_dict, is_sim, is_compress
def get_subdirectories(data_dir, target_dirs=None):
    if target_dirs is None:
        return [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, subdir))]
    else:
        return [os.path.join(data_dir, subdir) for subdir in target_dirs if os.path.isdir(os.path.join(data_dir, subdir))]

def get_files(dataset_dirs, robot_infor, dateset_type='train'):
    read_h5files = ReadH5Files(robot_infor)
    
    # target_dirs = ["put_the_red_pepper_and_place_it_into_bowl"]
    # target_dirs = ["open_the_pot", "grasp_brown_steamed_buns_in_the_pan",
    #                "put_the_yellow_pepper_and_place_it_into_bowl","put_the_red_pepper_and_place_it_into_bowl",
    #                "put_the_yellow_pepper_and_place_it_into_basket","put_the_red_pepper_and_place_it_into_basket",
    #                "close_the_pot"]
    # target_dirs = None
    # target_dirs = [
    #             "241022_side_pull_close_drawer_1","241022_side_pull_open_drawer_1"]
    target_dirs = ["place_in_bread_on_plate_2","pick_up_strawberry_in_bowl","open_cap_trash_can_1",
                   "241022_side_pull_close_drawer_1","241022_side_pull_open_drawer_1"]
    dataset_dirs = get_subdirectories(dataset_dirs,target_dirs)
    files = []

    for dataset_dir in dataset_dirs:
        dataset_dir = os.path.join(dataset_dir, 'success_episodes', dateset_type)
        for trajectory_id in sorted(os.listdir(dataset_dir)):
            trajectory_dir = os.path.join(dataset_dir, trajectory_id)
            file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
            try:
                _, control_dict, base_dict, _, is_compress = read_h5files.execute(file_path, camera_frame=2) # check at first time wash out short h5 data
                files.append(file_path)
            except Exception as e:
                print(f"Open {file_path} fail!")
                print(e)
    return files

def worldtopixel_point(traj,rgb, camera_intrinsics,camera_extrinsics):

        flange_position = traj[:,:3]
        flange_orientation = traj[:,3:]
        gripper_offset = np.array([0, 0, 0.15])
        gripper_positions = []
        # 对每个时刻进行计算
        for i in range(traj.shape[0]):
            # 提取每个时刻的法兰世界坐标和姿态
            flange_pos = flange_position[i]
            flange_orient = flange_orientation[i]
            
            # 将欧拉角转换为旋转矩阵
            rotation = Rotation.from_euler('xyz', flange_orient, degrees=False)  # 假设的欧拉角顺序
            rotation_matrix = rotation.as_matrix()

            # 计算末端夹爪的世界坐标
            gripper_pos = flange_pos + rotation_matrix @ gripper_offset

            # 存储结果
            gripper_positions.append(gripper_pos)
        gripper_positions = np.array(gripper_positions)
        # 转换到ee的世界坐标
        update_traj = gripper_positions[:,:3]

        # update_traj = traj[:,:3].copy()  # 保留原始数据
        
        # update_traj[:, 2] -= 0.17

        P_world_homogeneous = np.hstack([update_traj, np.ones((update_traj.shape[0], 1))])

        # 2. 使用外参矩阵将世界坐标转换为相机坐标
        R = camera_extrinsics[:3, :3]
        T = camera_extrinsics[:3, 3]
        R_inv = R.T
        T_inv = -R_inv @ T
        camera_extrinsics_inv = np.eye(4)
        camera_extrinsics_inv[:3, :3] = R_inv
        camera_extrinsics_inv[:3, 3] = T_inv
        camera_coordinates_homogeneous = (camera_extrinsics_inv @ P_world_homogeneous.T).T  # (4,)
        camera_coordinates = camera_coordinates_homogeneous[:, :3]
        # 计算像素坐标 u, v
        u = (camera_intrinsics[0, 0] * camera_coordinates[:, 0] / camera_coordinates[:, 2]) + camera_intrinsics[0, 2]
        v = (camera_intrinsics[1, 1] * camera_coordinates[:, 1] / camera_coordinates[:, 2]) + camera_intrinsics[1, 2]

        pixel_coordinates = np.stack((u, v), axis=-1) 
        pixel_coordinates = np.maximum(pixel_coordinates, 1)
        # 校验 pixel_coordinates 的合法性
        valid_mask = (pixel_coordinates[:, 0] >= 0) & (pixel_coordinates[:, 0] < rgb.shape[2]) & \
                    (pixel_coordinates[:, 1] >= 0) & (pixel_coordinates[:, 1] < rgb.shape[1])
        valid_pixel_coordinates = pixel_coordinates[valid_mask]
        
        # visualization 轨迹
        # rgb_vis = rgb.permute(1, 2, 0).numpy()
        # for (u, v) in valid_pixel_coordinates:
        #     cv2.circle(rgb_vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色点

        # cv2.imwrite("traj_predict/script/rgb_vis.png", rgb_vis)
        return valid_pixel_coordinates

def save_to_lmdb(output_dir, input_dir, dateset_type):

    robot_infor = {
        'camera_sensors': ['rgb_images'],
        'camera_names': ['camera_left', 'camera_right', 'camera_top'],
        'arms': ['puppet'],
        'controls': ['joint_position', 'end_effector'],
        'use_robot_base': False
    }
    files = get_files(input_dir, robot_infor, dateset_type)

    # 内参矩阵
    top_camera_intrinsics = np.array([
                                    [909.62, 0.0, 635.70],
                                    [0.0, 908.73, 364.50],
                                    [0.0, 0.0, 1.0]])

    top_camera_extrinsics = np.array([
                                    [-0.04640444,  0.91275018, -0.40587405,  1.05325108],
                                    [ 0.99843405,  0.0550883 ,  0.0097323 ,  0.11058065],
                                    [ 0.03124207, -0.40478685, -0.9138772 ,  0.98161176],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])

    left_camera_intrinsics = np.array([
                                    [608.41, 0.0, 318.07],
                                    [0.0, 608.31, 256.19],
                                    [0.0, 0.0, 1.0]])

    left_camera_extrinsics = np.array([
                                    [-0.90296094, -0.21165836,  0.37398166,  0.31437289],
                                    [-0.42258445,  0.59532919, -0.68337804,  1.02081076],
                                    [-0.07799952, -0.77510251, -0.62700254,  0.85824058],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    

    right_camera_intrinsics = np.array([
                                    [609.10, 0.0, 328.12],
                                    [0.0, 608.82, 247.01],
                                    [0.0, 0.0, 1.0]])
    right_camera_extrinsics = np.array([
                                    [ 0.49095751, -0.55381806,  0.67249259, -0.01004508],
                                    [-0.86493439, -0.21757396,  0.45227212, -0.60783651],
                                    [-0.10415959, -0.80370836, -0.58583586,  0.89650023],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])


    trial_files = files
    read_h5files = ReadH5Files(robot_infor)
    all_action_data = []
    all_episode_len = []

    trial_indices = []
    for id, filename in enumerate(files):
        try:
            _, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(filename, camera_frame=0)

            puppet_joint_position = control_dict['puppet']['joint_position'][:]
            action = puppet_joint_position
            action = action[:]


            for frame_id in range(action.shape[0]):
                trial_indices.append((id, frame_id))
            all_action_data.append(torch.from_numpy(action))
            all_episode_len.append(len(action))
        except Exception as e:
            print(e)
            print('filename:',filename)
    env = lmdb.open(output_dir, map_size=int(3e12), readonly=False, lock=False) # maximum size of memory map is 3TB

    
    with env.begin(write=True) as txn:
        if txn.get('cur_step'.encode()) is not None:
            cur_step = loads(txn.get('cur_step'.encode())) + 1
            cur_episode = loads(txn.get(f'cur_episode_{cur_step - 1}'.encode())) + 1
        else:
            cur_step = 0
            cur_episode = 0
        
        all_joint_position = []
        all_delta_joint_position = []
        for index, len_ep in enumerate(all_episode_len):
            print(f'{index/len(all_episode_len)}')
            trial_file = trial_files[index]
            with h5py.File(trial_file, 'r') as root:
                if 'language_raw' in root.keys():
                    inst = root['language_raw'][0].decode('utf-8')
                    print("dataset inst: ", inst)
                else:
                    # 根据文件名字定义
                    inst = Path(trial_file).parts[-6].replace('_', ' ')
                    print("inst: ", inst)
            # inst = "pick up the bread and put it on the plate" 
            txn.put(f'inst_{cur_episode}'.encode(), dumps(inst))
            with torch.no_grad():
                inst_token = clip.tokenize(inst)
                inst_emb = model_clip.encode_text(inst_token.cuda()).cpu()
            txn.put(f'inst_token_{cur_episode}'.encode(), dumps(inst_token[0]))
            txn.put(f'inst_emb_{cur_episode}'.encode(), dumps(inst_emb[0]))
            # threshold = 0.02
            threshold = -0.02 # 不起作用
            step_set = 5
            for start_ts in range(0, len_ep-1):
                trial_file = trial_files[index]

                top_camera_extrinsics_update = np.copy(top_camera_extrinsics)
                # 看是否需要补偿相机位置
                # top_camera_extrinsics_update[1, 3] += 0.12  #pixel y +为左方向
                # top_camera_extrinsics_update[0, 3] -= 0.03  #pixel x +为上方向
                # 图像
                image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(trial_file, camera_frame=start_ts, use_depth_image=False)
                # filter static action from begnning
                diff = np.abs(control_dict['puppet']['joint_position'][min(len_ep-1,start_ts + step_set)] - control_dict['puppet']['joint_position'][start_ts])
                if np.all(diff < threshold):
                    continue
                else:
                    frame = image_dict['rgb_images']
                    rgb_camera_left = torch.from_numpy(rearrange(frame['camera_left'], 'h w c -> c h w'))
                    txn.put(f'rgb_camera_left_{cur_step}'.encode(), dumps(encode_jpeg(rgb_camera_left)))
                    rgb_camera_right = torch.from_numpy(rearrange(frame['camera_right'], 'h w c -> c h w'))
                    txn.put(f'rgb_camera_right_{cur_step}'.encode(), dumps(encode_jpeg(rgb_camera_right)))
                    rgb_camera_top = torch.from_numpy(rearrange(frame['camera_top'], 'h w c -> c h w'))
                    txn.put(f'rgb_camera_top_{cur_step}'.encode(), dumps(encode_jpeg(rgb_camera_top)))

                    # 2d_trajectory
                    traj = control_dict['puppet']['end_effector'][start_ts:]
                    
                    traj_2d_top = worldtopixel_point(traj,rgb_camera_top, top_camera_intrinsics,top_camera_extrinsics_update)
                    traj_2d_top_tensor = torch.tensor(traj_2d_top)
                    txn.put(f'traj_2d_top_{cur_step}'.encode(), dumps(traj_2d_top_tensor))
                    # visualization
                    # rgb_vis = rgb_camera_top.permute(1, 2, 0).numpy()
                    # for (u, v) in traj_2d_top_tensor:
                    #     cv2.circle(rgb_vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色点
                    # cv2.imwrite(f"tools/visualization/rgb_vis_top_{index}_{start_ts}.png", rgb_vis)
                    
                    traj_2d_left = worldtopixel_point(traj,rgb_camera_left, left_camera_intrinsics,left_camera_extrinsics)
                    traj_2d_left_tensor = torch.tensor(traj_2d_left)
                    txn.put(f'traj_2d_left_{cur_step}'.encode(), dumps(traj_2d_left_tensor))
                    # rgb_vis = rgb_camera_left.permute(1, 2, 0).numpy()
                    # for (u, v) in traj_2d_left_tensor:
                    #     cv2.circle(rgb_vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色点
                    # cv2.imwrite(f"traj_predict/script/visualization/dataset/rgb_vis_left_{index}_{start_ts}.png", rgb_vis)

                    traj_2d_right = worldtopixel_point(traj,rgb_camera_right, right_camera_intrinsics,right_camera_extrinsics)
                    traj_2d_right_tensor = torch.tensor(traj_2d_right)
                    txn.put(f'traj_2d_right_{cur_step}'.encode(), dumps(traj_2d_right_tensor))
                    # rgb_vis = rgb_camera_right.permute(1, 2, 0).numpy()
                    # for (u, v) in traj_2d_right_tensor:
                    #     cv2.circle(rgb_vis, (int(u), int(v)), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色点
                    # cv2.imwrite(f"tools/visualization/rgb_vis_right_{index}_{start_ts}.png", rgb_vis)

                    
                    if start_ts == 0:# 起始点存储
                        txn.put(f'traj_2d_top_init_{cur_episode}'.encode(), dumps(traj_2d_top_tensor))
                        txn.put(f'traj_2d_left_init_{cur_episode}'.encode(), dumps(traj_2d_left_tensor))
                        txn.put(f'traj_2d_right_init_{cur_episode}'.encode(), dumps(traj_2d_right_tensor))

                    txn.put('cur_step'.encode(), dumps(cur_step))
                    txn.put(f'cur_episode_{cur_step}'.encode(), dumps(cur_episode))
                    txn.put(f'done_{cur_step}'.encode(), dumps(False))
                    # 对应的action
                    joint_position = control_dict['puppet']['joint_position'][start_ts]
                    txn.put(f'joint_position_{cur_step}'.encode(), dumps(torch.from_numpy(joint_position)))
                    all_joint_position.append(torch.from_numpy(joint_position).unsqueeze(0))

                    delta_joint_position = control_dict['puppet']['joint_position'][start_ts + 1] - control_dict['puppet']['joint_position'][start_ts]
                    txn.put(f'delta_joint_position_{cur_step}'.encode(), dumps(torch.from_numpy(delta_joint_position)))
                    all_delta_joint_position.append(torch.from_numpy(delta_joint_position).unsqueeze(0))
                    end_effector = control_dict['puppet']['end_effector'][start_ts]
                    txn.put(f'end_effector_{cur_step}'.encode(), dumps(torch.from_numpy(end_effector)))
                    delta_end_effector = control_dict['puppet']['end_effector'][start_ts + 1] - control_dict['puppet']['end_effector'][start_ts]
                    txn.put(f'delta_end_effector_{cur_step}'.encode(), dumps(torch.from_numpy(delta_end_effector)))

                    cur_step += 1
            txn.put(f'done_{cur_step-1}'.encode(), dumps(True))
            cur_episode += 1
        # stats joint
        all_joint_position = torch.cat(all_joint_position, dim=0)
        all_joint_position_mean = all_joint_position.mean(dim=[0]).float()
        all_joint_position_std = all_joint_position.std(dim=[0]).float()
        all_joint_position_std = torch.clip(all_joint_position_std, 1e-2, np.inf)  # clipping
        all_joint_position_min = all_joint_position.min(dim=0).values.float()
        all_joint_position_max = all_joint_position.max(dim=0).values.float()
        # stats delta joint
        all_delta_joint_position = torch.cat(all_delta_joint_position, dim=0)
        all_delta_joint_position_mean = all_delta_joint_position.mean(dim=[0]).float()
        all_delta_joint_position_std = all_delta_joint_position.std(dim=[0]).float()
        all_delta_joint_position_std = torch.clip(all_delta_joint_position_std, 1e-2, np.inf)  # clipping

        all_delta_joint_position_min = all_delta_joint_position.min(dim=0).values.float()
        all_delta_joint_position_max = all_delta_joint_position.max(dim=0).values.float()
        eps = 0.0001
        stats = {"joint_mean": all_joint_position_mean, "joint_std": all_joint_position_std,
                "joint_min": all_joint_position_min - eps, "joint_max": all_joint_position_max + eps,
                "delta_joint_mean": all_delta_joint_position_mean, "delta_joint_std": all_delta_joint_position_std,
                "delta_joint_min": all_delta_joint_position_min - eps, "delta_joint_max": all_delta_joint_position_max + eps}
        txn.put(b'stats', dumps(stats))

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer CALVIN dataset to lmdb format.")
    # parser.add_argument("--input_dir", default='/nfsroot/DATA/users/embodied_ai/gary/pick_bread_plate/241120_pick_bread_plate_1', type=str, help="Original dataset directory.") # debug
    # parser.add_argument("--output_dir", default='/nfsroot/DATA/IL_Research/datasets/dual_franka/real_franka_1/h5_data_1/241016_remove_blue_cube_pink_1/success_episodes_lmdb_1', type=str, help="Target dataset directory.") #debug
    # parser.add_argument("--input_dir", default='/nfsroot/DATA/users/embodied_ai/gary/pick_bread_plate/', type=str, help="Original dataset directory.")
    # parser.add_argument("--output_dir", default='/bamboo_dir/pick_bread_plate/success_episodes_lmdb_1', type=str, help="Target dataset directory.")
    # parser.add_argument("--input_dir", default='/media/data/benchmark2_0_eval/', type=str, help="Original dataset directory.")
    # parser.add_argument("--output_dir", default='/media/users/bamboo/dataset/lmdb/evaluation_1/', type=str, help="Target dataset directory.")
    
    parser.add_argument("--input_dir", default='/media/data/benchmark1_0_release_pretrain/h5_franka_3rgb', type=str, help="Original dataset directory.")
    parser.add_argument("--output_dir", default='/media/users/bamboo/dataset/lmdb/benchmark1_0_release_pretrain_franka_5tasks/', type=str, help="Target dataset directory.")
    
    args = parser.parse_args()
    model_clip, _ = clip.load('/media/users/bamboo/PretrainModel/clip/ViT-B-32.pt', device='cuda:0')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_to_lmdb(args.output_dir, args.input_dir, 'train')
    print("training data finished")
    save_to_lmdb(args.output_dir, args.input_dir, 'val')
    print("val data finished")

    
# python traj_predict/script/trajectrory_sampling.py --input_dir=/home/DATASET_PUBLIC/calvin/task_ABC_D --output=/home/DATASET_PUBLIC/calvin/task_ABC_D/calvin_lmdb_V1
# python python traj_predict/script/trajectrory_sampling.py --input_dir=/home/DATASET_PUBLIC/calvin/task_D_D/ --output=/home/DATASET_PUBLIC/calvin/task_D_D/calvin_lmdb_V1

