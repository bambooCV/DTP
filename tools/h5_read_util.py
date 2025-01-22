import h5py
import os
import cv2
import numpy as np
from collections import defaultdict
import gc
import psutil
import torch

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
            is_compress = True
            lang_embed = None
            # print(f"root keys: {root.keys()}")
            if 'language_distilbert' in root:
                lang_embed = root['language_distilbert'][:]
                lang_embed = torch.from_numpy(lang_embed).float().squeeze()
            else:
                # dummy value
                lang_embed = torch.zeros(1)
            control_dict['language_distilbert'] = lang_embed

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
                            # x = self.camera_sensors[0]
                            # y = cam_name
                            # z = camera_frame
                            # print(f"x: {x}")
                            # print(f"y: {y}")
                            # print(f"z: {z}")
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


if __name__ == '__main__':

    # robot_infor = {'camera_names': ['camera_left', 'camera_right'],
    #                'camera_sensors': ['rgb_images'],
    #                'arms': ['puppet'],
    #                'controls': ['joint_position',  'end_effector']}
    franka_3rgb = {
            'camera_sensors': ['rgb_images'],
            'camera_names': ['camera_left', 'camera_right', 'camera_top','camera_wrist'],
            'arms': ['puppet'],
            'controls': ['joint_position', 'end_effector'],
            'use_robot_base': False
        }
    robot_infor_franka_1rgb = {'camera_names': ['camera_top'],
                   'camera_sensors': ['rgb_images'],
                   'arms': ['puppet'],
                   'controls': ['joint_position',  'end_effector'],
                    'use_robot_base': False}
    robot_infor_ur_1rgb = {
                    'camera_sensors': ['rgb_images'],
                    'camera_names': ['camera_top'],
                    'arms': ['puppet'],
                    'controls': ['joint_position', 'end_effector'],
                    'use_robot_base': False
    }



    read_h5files = ReadH5Files(franka_3rgb)
    file_path = "/media/data/benchmark2_0_eval/open_the_pot/success_episodes/train/0106_102610/data/trajectory.hdf5"
    # file_path = '/media/data/benchmark2_0_eval/put_the_red_pepper_and_place_it_into_basket/success_episodes/train/0106_153800/data/trajectory.hdf5'
    start_ts = 20
    image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(file_path, camera_frame=start_ts)
    frame = image_dict['rgb_images']
    
    cv2.imwrite(f"tools/visualization/h5_vis.png", frame['camera_top'])
    
    print("fsc test")




