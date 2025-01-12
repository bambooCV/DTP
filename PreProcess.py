from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Resize
from torchvision import transforms
def RandomShiftsAug_2D(x, pad):
    x = x.float()
    b, t, c, h, w = x.size() # torch.Size([10, 10, 3, 200, 200])
    assert h == w
    x = x.view(b*t, c, h, w)  # reshape x to [B*T, C, H, W] torch.Size([100, 3, 200, 200])
    padding = tuple([pad] * 4) # (10,10,10,10)
    x = F.pad(x, padding, "replicate") # 上下左右各10 torch.Size([100, 3, 220, 220])
    h_pad, w_pad = h + 2*pad, w + 2*pad  #220，220 calculate the height and width after padding
    eps = 1.0 / (h_pad)  # 1/220
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h_pad, device=x.device, dtype=x.dtype)[:h] # 以步长eps 范围是[-1,1] 长度为220，而后取前200个点
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2) # [200,200,1]
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)#torch.Size([200, 200, 2])
    base_grid = base_grid.unsqueeze(0).repeat(b*t, 1, 1, 1)#torch.Size([100, 200, 200, 2]) (x,y)原始图像每个像素的位置，且位置呗标准化到[-1，1]

    shift = torch.randint(0, 2 * pad + 1, size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype) # x,y 方向的随机平移量
    # shift = torch.zeros(size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype)# 这里设定为0，即不做平移
    shift = shift.repeat(1, t, 1, 1, 1)  # repeat the shift for each image in the sequence 同一个batch里的squence 同一个时间步的shift
    shift = shift.view(b*t, 1, 1, 2)  # reshape shift to match the size of base_grid torch.Size([100, 1, 1, 2])
    shift *= 2.0 / (h_pad)

    grid = base_grid + shift
    output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    output = output.view(b, t, c, h, w)  # reshape output back to [B, T, C, H, W]
    shift = shift.view(b, t, 1, 2)  # reshape shift back to [B, T, 1, 2]
    return output,shift
def RandomShiftsAug(x, pad):
    x = x.float()
    b, t, c, h, w = x.size()
    assert h == w
    x = x.view(b*t, c, h, w)  # reshape x to [B*T, C, H, W]
    padding = tuple([pad] * 4)
    x = F.pad(x, padding, "replicate")
    h_pad, w_pad = h + 2*pad, w + 2*pad  # calculate the height and width after padding
    eps = 1.0 / (h_pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h_pad, device=x.device, dtype=x.dtype)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(b*t, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift = shift.repeat(1, t, 1, 1, 1)  # repeat the shift for each image in the sequence
    shift = shift.view(b*t, 1, 1, 2)  # reshape shift to match the size of base_grid
    shift *= 2.0 / (h_pad)

    grid = base_grid + shift
    output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    output = output.view(b, t, c, h, w)  # reshape output back to [B, T, C, H, W]
    return output
def shifts_2d_action_to_Aug(action_2d, shift, pad, w, h):
    # 获取张量的形状
    b, seq, t, dim = action_2d.shape

    norm_x_o = action_2d[..., 0] + pad
    norm_y_o = action_2d[..., 1] + pad
    shift_o = shift * (w + 2*pad) / 2
    # 获取shift值
    shift_x = shift_o[..., 0] 
    shift_y = shift_o[..., 1]

    # 计算新图像中对应的标准化坐标
    new_norm_x_o = norm_x_o - shift_x 
    new_norm_y_o = norm_y_o - shift_y

    # 合并新的坐标
    augmented_action_2d = torch.stack([new_norm_x_o, new_norm_y_o], dim=-1)

    return augmented_action_2d

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
class PreProcess(): 
    def __init__(
            self,
            rgb_static_pad,
            rgb_gripper_pad,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            device,
        ):
       
        self.resize = Resize(rgb_shape, interpolation=Image.BICUBIC, antialias=True).to(device)
        self.rgb_static_pad = rgb_static_pad
        self.rgb_gripper_pad = rgb_gripper_pad
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 1, -1, 1, 1)
    
    def rgb_process(self, rgb_static, rgb_gripper, train=False):
        rgb_static = rgb_static.float()*(1/255.)
        rgb_gripper = rgb_gripper.float()*(1/255.)
        if train:
            rgb_static = RandomShiftsAug(rgb_static, self.rgb_static_pad)
            rgb_gripper = RandomShiftsAug(rgb_gripper, self.rgb_gripper_pad)
        rgb_static = self.resize(rgb_static)
        rgb_gripper = self.resize(rgb_gripper)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_static = (rgb_static - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_gripper = (rgb_gripper - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_static, rgb_gripper

    def rgb_recovery(self, rgb_static):
        rgb_static = (rgb_static * (self.rgb_std + 1e-6)) + self.rgb_mean
        rgb_static = rgb_static.clamp(0, 1)
        rgb_static = (rgb_static*255.).byte()
        return rgb_static

class Real_Robot_PreProcess(): 
    def __init__(
            self,
            rgb_static_pad,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            device,
        ):
       
        self.resize = Resize(rgb_shape, interpolation=Image.BICUBIC, antialias=True).to(device)
        self.rgb_static_pad = rgb_static_pad
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 1, -1, 1, 1)
    
    def rgb_process(self, rgb_camera_left, rgb_camera_right, rgb_camera_top, train=False):

        rgb_camera_left = rgb_camera_left.float()*(1/255.)
        rgb_camera_right = rgb_camera_right.float()*(1/255.)
        rgb_camera_top = rgb_camera_top.float()*(1/255.)
        
        rgb_camera_left = self.resize(rgb_camera_left)
        rgb_camera_right = self.resize(rgb_camera_right)
        rgb_camera_top = self.resize(rgb_camera_top)

        # import cv2
        # import numpy as np
        # image =  rgb_camera_top[0][0] 
        # rgb_vis = image.permute(1, 2, 0).cpu().numpy().copy()
        # rgb_vis = (rgb_vis * 255).astype(np.uint8)  # 乘回 255，并转换为 uint8
        # cv2.imwrite("tmp.png", rgb_vis)  
        if train:
            rgb_camera_left = RandomShiftsAug(rgb_camera_left, self.rgb_static_pad)
            rgb_camera_right = RandomShiftsAug(rgb_camera_right, self.rgb_static_pad)
            rgb_camera_top = RandomShiftsAug(rgb_camera_top, self.rgb_static_pad)
        # import cv2
        # import numpy as np
        # image =  rgb_camera_top[0][0] 
        # rgb_vis = image.permute(1, 2, 0).cpu().numpy().copy()
        # rgb_vis = (rgb_vis * 255).astype(np.uint8)  # 乘回 255，并转换为 uint8
        # cv2.imwrite("tmp_aug.png", rgb_vis)  


        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_camera_left = (rgb_camera_left - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_camera_right = (rgb_camera_right - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_camera_top = (rgb_camera_top - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_camera_left, rgb_camera_right, rgb_camera_top

    def rgb_recovery(self, rgb_static):
        rgb_static = (rgb_static * (self.rgb_std + 1e-6)) + self.rgb_mean
        rgb_static = rgb_static.clamp(0, 1)
        rgb_static = (rgb_static*255.).byte()
        return rgb_static
    
class Real_Robot_2D_PreProcess(): 
    def __init__(
            self,
            rgb_static_pad,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            device,
        ):
       
        self.resize = Resize(rgb_shape, interpolation=Image.BICUBIC, antialias=True).to(device)
        self.rgb_static_pad = rgb_static_pad
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 1, -1, 1, 1)
    
    def rgb_process(self, rgb_camera_left, rgb_camera_right, rgb_camera_top, action_2d_top, action_2d_top_rest,train=False):

        rgb_camera_left = rgb_camera_left.float()*(1/255.)
        rgb_camera_right = rgb_camera_right.float()*(1/255.)
        rgb_camera_top = rgb_camera_top.float()*(1/255.)
        
        rgb_camera_left = self.resize(rgb_camera_left)
        rgb_camera_right = self.resize(rgb_camera_right)
        rgb_camera_top = self.resize(rgb_camera_top)
        new_action_2d_top = resize_points(action_2d_top.clone(), (640,640), (224,224))
        new_action_2d_top_rest = resize_points(action_2d_top_rest.clone(), (640,640), (224,224))

        if train:
            rgb_camera_left,_ = RandomShiftsAug_2D(rgb_camera_left, self.rgb_static_pad)
            rgb_camera_right,_ = RandomShiftsAug_2D(rgb_camera_right, self.rgb_static_pad)
            # # vis
            # import numpy as np
            # import cv2
            # image =  rgb_camera_top[0][0] 
            # action = new_action_2d_top_rest[0][0] 
            # rgb_vis = image.permute(1, 2, 0).cpu().numpy().copy()
            # rgb_vis = (rgb_vis * 255).astype(np.uint8)  # 乘回 255，并转换为 uint8
            # for index, point_2d in enumerate(action):
            #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=2, color=(255, 255, 255), thickness=-1)
            #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=1, color=(225,206,135), thickness=-1)
            # cv2.imwrite("tmp.png", rgb_vis)  

            rgb_camera_top,top_shift = RandomShiftsAug_2D(rgb_camera_top, self.rgb_static_pad)
            new_action_2d_top = shifts_2d_action_to_Aug(new_action_2d_top, top_shift,self.rgb_static_pad,rgb_camera_top.shape[-1],rgb_camera_top.shape[-2])
            new_action_2d_top = torch.clamp(new_action_2d_top, 0, 224)
            new_action_2d_top_rest = shifts_2d_action_to_Aug(new_action_2d_top_rest, top_shift,self.rgb_static_pad,rgb_camera_top.shape[-1],rgb_camera_top.shape[-2])
            new_action_2d_top_rest = torch.clamp(new_action_2d_top_rest, 0, 224)

            # import numpy as np
            # import cv2
            # image =  rgb_camera_top[0][0] 
            # action = new_action_2d_top_rest[0][0] 
            # rgb_vis = image.permute(1, 2, 0).cpu().numpy().copy()
            # rgb_vis = (rgb_vis * 255).astype(np.uint8)  # 乘回 255，并转换为 uint8
            # for index, point_2d in enumerate(action):
            #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=2, color=(255, 255, 255), thickness=-1)
            #     cv2.circle(rgb_vis, tuple(point_2d.int().tolist()), radius=1, color=(225,206,135), thickness=-1)
            # cv2.imwrite("tmp_aug.png", rgb_vis)  

        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_camera_left = (rgb_camera_left - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_camera_right = (rgb_camera_right - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_camera_top = (rgb_camera_top - self.rgb_mean) / (self.rgb_std + 1e-6)
        
        return rgb_camera_left, rgb_camera_right, rgb_camera_top, new_action_2d_top,new_action_2d_top_rest

    def rgb_recovery(self, rgb_static):
        rgb_static = (rgb_static * (self.rgb_std + 1e-6)) + self.rgb_mean
        rgb_static = rgb_static.clamp(0, 1)
        rgb_static = (rgb_static*255.).byte()
        return rgb_static