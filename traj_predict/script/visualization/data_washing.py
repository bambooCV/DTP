
import os
import cv2
import numpy as np
import h5py
from collections import defaultdict
import shutil
def get_files(dataset_dir, robot_infor):
    read_h5files = ReadH5Files(robot_infor)
    success_episodes_dir = os.path.join(dataset_dir, 'success_episodes')

    # 判断目录是否存在
    if os.path.exists(success_episodes_dir):
        dataset_dir = os.path.join(dataset_dir, 'success_episodes')
        files = []
        for trajectory_id in sorted(os.listdir(dataset_dir)):
            trajectory_dir = os.path.join(dataset_dir, trajectory_id)
            file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
            try:
                _, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(file_path, camera_frame=2)
                files.append(file_path)
            except Exception as e:
                print(e)
    else:
        # 获得子文件夹内所有文件
        from datetime import datetime
        subfolders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
        
        sorted_subfolders = sorted(subfolders, key=lambda x: datetime.strptime(x[:6], '%y%m%d'))
        sorted_subfolders = sorted_subfolders[:9] # 只用0-8 
        files = []
        for sorted_subfolder in sorted_subfolders:
            single_dataset_dir = os.path.join(dataset_dir,sorted_subfolder,'success_episodes')
            for trajectory_id in sorted(os.listdir(single_dataset_dir)):
                trajectory_dir = os.path.join(single_dataset_dir, trajectory_id)
                file_path = os.path.join(trajectory_dir, 'data/trajectory.hdf5')
                try:
                    _, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(file_path, camera_frame=2)
                    files.append(file_path)
                except Exception as e:
                    print(e)
    print(len(files))
    return files
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
            is_compress = root.attrs['compress']
            # print(f"is_compress: {is_compress}")
            # select camera frame id
            for cam_name in self.camera_names:
                if is_compress:
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
                        image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
                    # print(f"decode_rgb size: {decode_rgb.shape}")

                else:
                    if camera_frame:
                        if use_depth_image:
                            image_dict[self.camera_sensors[0]][cam_name] = root[
                                'observations'][self.camera_sensors[0]][cam_name][camera_frame]
                            image_dict[self.camera_sensors[1]][cam_name] = root[
                                'observations'][self.camera_sensors[1]][cam_name][camera_frame]
                        else:
                            image_dict[self.camera_sensors[0]][cam_name] = root[
                                'observations'][self.camera_sensors[0]][cam_name][camera_frame]
                    else:
                        if use_depth_image:
                            image_dict[self.camera_sensors[0]][cam_name] = root[
                               'observations'][self.camera_sensors[0]][cam_name][()]
                            image_dict[self.camera_sensors[1]][cam_name] = root[
                               'observations'][self.camera_sensors[1]][cam_name][()]
                        else:
                            image_dict[self.camera_sensors[0]][cam_name] = root[
                                'observations'][self.camera_sensors[0]][cam_name][()]

            # print('image_dict:',image_dict)
            for arm_name in self.arms:
                for control in self.robot_infor:
                    if control_frame:
                        control_dict[arm_name][control] = root[arm_name][control][control_frame]
                    else:
                        control_dict[arm_name][control] = root[arm_name][control][()]
            # print('infor_dict:',infor_dict)
        # gc.collect()
        return image_dict, control_dict, base_dict, is_sim, is_compress

# config
robot_infor = {'camera_names': ['camera_left', 'camera_right', 'camera_top'],
            'camera_sensors': ['rgb_images'],
            'arms': ['puppet'],
            'controls': ['joint_position',  'end_effector']}
input_dir = "/home/bamboofan/EmbodiedAI/dataset/pick_bread_plate"
save_dir = "/home/bamboofan/EmbodiedAI/dataset/pick_bread_plate_washing/"
files = get_files(input_dir, robot_infor) # 读取所有file路径
# 把文件读入
read_h5files = ReadH5Files(robot_infor)

for id, filename in enumerate(files):
    try:
        image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(filename, camera_frame=0)

        rgb_vis = image_dict['rgb_images']['camera_top']
        cv2.imshow('rgb_vis', rgb_vis)
        print(id)
        key = cv2.waitKey(0) & 0xFF  # 读取按键

        if key == ord('1'):  # 检查是否按下 '1' 键 
            save_path = filename.replace('/pick_bread_plate/', '/pick_bread_plate_washing/bread_right/')
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)  # 确保目标目录存在
            shutil.copy2(filename, save_path)  # 复制文件
            print(f"File successfully saved to {save_path}")
        elif key == ord('2'):  # 检查是否按下 '2' 键
            save_path = filename.replace('/pick_bread_plate/', '/pick_bread_plate_washing/bread_left/')
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)  # 确保目标目录存在
            shutil.copy2(filename, save_path)  # 复制文件
            print(f"File successfully saved to {save_path}")
        else:
            print("No action taken.")


    except Exception as e:
        print(e)
        print('filename:',filename)

print("fsc test")