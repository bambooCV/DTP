import warnings

import os
import sys
sys.path.append('..')

sys.path.append('/home/ps/Dev/bamboo/inrocs_online_inference/diffusion-trajectory-guided-policy')
sys.path.append('/home/ps/Dev/bamboo/inrocs_online_inference')
import threading
import time
from pathlib import Path
from einops import rearrange

import torch
import cv2
import json

import numpy as np

from Real_Robot_Main_Inference_dev import DTP_Evaluation
# from pynput import keyboard


# from robot_env.franka_env import robot_env

import torch.nn.functional as F

# from leader.agent import LeaderAgent, BiLeaderAgent
# print('aaa')
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
import clip
preparing = True


import warnings
import datetime
import os
import sys


import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt

import hydra
import torch
import cv2
from skimage.transform import resize



import numpy as np
import tqdm
import tyro
from pynput import keyboard

from collector.format_obs import save_frame
from robot_env.franka_env import robot_env

from models.real_robot_gr1 import GR1 
import models.vision_transformer as vits
from PreProcess import Real_Robot_PreProcess

preparing = False





def on_press(key):
    global preparing
    try:
        if key == keyboard.Key.enter:
            preparing = False

    except AttributeError:
        pass


def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)



def main_DTP():
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = json.load(open('diffusion-trajectory-guided-policy/Real_Robot_configs_inference_dev.json'))
    DTP_Eva = DTP_Evaluation(cfg)
    print('DTP model loaded')
    
    listener_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    listener_thread.start()

    # going to start position
    print("Going to start position")

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
    os.system("espeak start")

    # warm up
    obs = robot_env.get_obs()

    ###
    print("enter enter to go")
    global preparing
    while preparing:
        ...
    preparing = True
    action_mean = torch.tensor([ 0.0224,  0.2196, -0.0879, -1.6439, -0.0494,  1.8743, -0.0359,  0.1230])
    action_std = torch.tensor([0.1829, 0.2051, 0.1402, 0.1658, 0.0625, 0.2278, 0.2456, 0.1852])
    stats = {"action_mean": action_mean, "action_std": action_std}
    obs = robot_env.get_obs()

    DTP_Eva.reset()
    for step in range(100000) :
        action = DTP_Eva.inference(obs,stats)
        print(action)
        obs = robot_env.step(action)
        
        # data = prepare_inference_obs(obs, norm_stats, sentence_encoder, cfg.num_queries)


    # fake_left_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # _, fake_left_image = cv2.imencode('.jpg', fake_left_image)
    # fake_right_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # _, fake_right_image = cv2.imencode('.jpg', fake_right_image)
    # fake_top_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # _, fake_top_image = cv2.imencode('.jpg', fake_top_image)

    # fake_obs = {
    #     'images': {
    #         'left': fake_left_image,
    #         'right':fake_right_image,  
    #         'top':fake_top_image,
    #     },
    #     'qpos': torch.randn(8,),
    #     'goal': "remove blue cube from pink cube",
    # }

    # action = DTP_Eva.inference(fake_obs)
    # print(action)
if __name__ == '__main__':
    main_DTP()