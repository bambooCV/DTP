import numpy as np

import cv2
from scipy.spatial.transform import Rotation 
import open3d as o3d
def depth_to_point_cloud(depth_image, rgb_image, camera_intrinsics):
    """
    将深度图和 RGB 图像转换为点云坐标。
    :param depth_image: 深度图 (H, W)，单位通常是米或毫米
    :param rgb_image: RGB 图像 (H, W, 3)
    :param camera_intrinsics: 相机内参矩阵 (3x3)
    :return: 点云 (N, 6)，每行包含 (X, Y, Z, R, G, B)
    """
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))  # 像素网格

    # 将深度和像素位置投影到相机坐标系
    X = (u - cx) * depth_image / fx
    Y = (v - cy) * depth_image / fy
    Z = depth_image

    # 展平坐标
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)  # (N, 3)

    # 如果有 RGB 图像，为点云附加颜色信息
    if rgb_image is not None:
        colors = rgb_image.reshape(-1, 3)  # (N, 3)
        points = np.hstack((points, colors))  # (N, 6)

    return points

def get_point_cloud_at_pixel(pixel_coords, depth_image, camera_intrinsics, rgb_image=None):
    """
    获取指定像素点在点云中的 (X, Y, Z) 坐标（以及颜色）。
    :param pixel_coords: 像素点坐标 (u, v)
    :param depth_image: 深度图 (H, W)
    :param camera_intrinsics: 相机内参矩阵 (3x3)
    :param rgb_image: RGB 图像 (H, W, 3)，可选
    :return: (X, Y, Z) 或 (X, Y, Z, R, G, B)
    """
    u, v = pixel_coords
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # 获取深度值
    Z = depth_image[v, u]  # 注意：图像索引是 (v, u)

    # 如果深度值无效（为 0 或 NaN），返回 None
    if Z <= 0 or np.isnan(Z):
        return None

    # 计算相机坐标系下的 3D 点
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    if rgb_image is not None:
        # 获取对应像素的 RGB 值
        R, G, B = rgb_image[v, u]
        return (X, Y, Z, R, G, B)
    else:
        return (X, Y, Z)
direction = 'top'
# index = 0
# pixel_coords = (285,38) #0
index = 88
pixel_coords = (11,11) 
pixel_coords = (62,112) 
# pixel_coords = (373,194)
# 读取 .npy 文件
file_path = f'traj_predict/script/visualization/depth/camera_{direction}_{index}.npy'  # 替换为实际文件路径
data = np.load(file_path)/1000
depth_image = data
image_path = f"traj_predict/script/visualization/rgb/camera_{direction}_{index}.jpg"  # 替换为你的图像路径
rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
file_path_EE = 'traj_predict/script/visualization/end_effector.txt'

# 打开文件并读取第5行
with open(file_path_EE, 'r') as f:
    lines = f.readlines()
    
    # 假设读取的是第0行（可以改为读取其他行，0索引）
    line = lines[index].strip()  # 使用 strip() 去掉多余的空格和换行符
    print("读取的行内容:", repr(line))  # 使用 repr 打印行，查看不可见字符
    
    # 去除方括号并替换逗号为空格
    line = line.strip('[]')  # 去除方括号
    line = line.replace(',', ' ')  # 将逗号替换为空格
    
    # 将该行数据按空格分割
    values = line.split()
    
    # 检查每个值是否可以转换为 float
    for val in values:
        try:
            float_val = float(val)
        except ValueError:
            print(f"无法将值转换为浮点数: {val}")
            continue
    
    # 试图将所有值转换为 NumPy 数组
    try:
        world_coordinates = np.array([float(val) for val in values])
        print("转换为NumPy数组的world_coordinates:", world_coordinates)
    except ValueError as e:
        print(f"错误: {e}")

# world_coordinates = np.array([0.6306840181350708, -0.07409526407718658, 0.5421301126480103, 3.1109101805269916, 0.0031032312538119555, 0.27285664605186016]) 
top_camera_intrinsics = np.array([
                                [909.62, 0.0, 635.70],
                                [0.0, 908.73, 364.50],
                                [0.0, 0.0, 1.0]])

top_camera_extrinsics = np.array([
                                [-0.04640444,  0.91275018, -0.40587405,  1.05325108],
                                [ 0.99843405,  0.0550883 ,  0.0097323 ,  0.11058065],
                                [ 0.03124207, -0.40478685, -0.9138772 ,  0.98161176],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic = np.array([1.0532510813527192, 0.11058064969153436, 0.9816117632129159,
                    -2.7246307295250696, -0.031247157971078243, 1.617240122129939])


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

if direction == 'top':
    camera_intrinsics = top_camera_intrinsics
    camera_extrinsics = top_camera_extrinsics
elif direction == 'left':
    camera_intrinsics = left_camera_intrinsics
    camera_extrinsics = left_camera_extrinsics
else:
    camera_intrinsics = right_camera_intrinsics
    camera_extrinsics = right_camera_extrinsics

all_points = depth_to_point_cloud(depth_image, rgb_image, camera_intrinsics)

point_cloud_data = all_points[:, :3]
colors = all_points[:, 3:] / 255.0  # 假设颜色在 0-255 范围内，需要归一化到 0-1
camera_point = get_point_cloud_at_pixel(pixel_coords, depth_image, camera_intrinsics, rgb_image)
camera_point = np.array(camera_point[:3])  # 确保只取 (X, Y, Z)
# 创建 Open3D 的点云对象
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
# if colors.shape[1] == 3:
#     pcd.colors = o3d.utility.Vector3dVector(colors)

# sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # 设置球体半径来控制大小
# sphere.translate(camera_point)  # 将球体移动到目标点
# sphere.paint_uniform_color([1, 0, 0])  # 将球体涂成红色以高亮显示
# o3d.visualization.draw_geometries([pcd, sphere])

#--------------------------------------------3D point cloud good in camera coordinate-------------------------------------------#

# 将点云通过外参转到armbase坐标系
# 将点云数据转换为齐次坐标，添加一列1
ones_column = np.ones((point_cloud_data.shape[0], 1))
point_cloud_homogeneous = np.hstack((point_cloud_data, ones_column))  # (N, 4)
transformed_points_homogeneous = (camera_extrinsics @ point_cloud_homogeneous.T).T  # (N, 4)
transformed_points = transformed_points_homogeneous[:, :3]  # (N, 3)
camera_point_homogeneous = np.append(camera_point[:3], 1)  # 转换为齐次坐标 (4,)
transformed_camera_point_homogeneous = camera_extrinsics @ camera_point_homogeneous  # (4,)
transformed_camera_point = transformed_camera_point_homogeneous[:3]  # 取前 3 个值 (X, Y, Z)

# world ground truth vis
if 1:
    world_coordinates = world_coordinates[:3]
    # world_coordinates[2] = world_coordinates[2] # - 0.11  #  end-effector
else:
    # rotation
    flange_position = world_coordinates[:3]
    flange_orientation = world_coordinates[3:]
    gripper_offset = np.array([-0.04, 0.0, 0.15])
    # gripper_offset = np.array([0.0, 0, 0.15])
    flange_pos = flange_position
    flange_orient = flange_orientation
    rotation = Rotation.from_euler('xyz',flange_orient,degrees=False)  # 假设的欧拉角顺序
    rotation_matrix = rotation.as_matrix()
    gripper_pos = flange_pos + rotation_matrix @ gripper_offset
    world_coordinates = gripper_pos






# 创建转换后的点云对象
transformed_pcd = o3d.geometry.PointCloud()
transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # 设置球体半径来控制大小
sphere.translate(transformed_camera_point)  # 将球体移动到目标点
sphere.paint_uniform_color([1, 0, 0])  # 将球体涂成红色以高亮显示
# 创建一个球体来标记world point
sphere_world_groundtruth = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 设置球体半径来控制大小
sphere_world_groundtruth.translate(world_coordinates)  # 将球体移动到目标点
sphere_world_groundtruth.paint_uniform_color([0, 1, 0])  
# 可视化转换后的点云和高亮球体
o3d.visualization.draw_geometries([transformed_pcd, sphere,sphere_world_groundtruth]) # camera->world
print(f"transformed_world_point: {transformed_camera_point}, world_groudtruth:{world_coordinates} ")

#--------------------------------------------3D point cloud good in world coordinate-------------------------------------------#


# inverse: world->camera->pixel
# 已知的数据

# 1. 将世界坐标转换为齐次坐标

world_coordinates_homogeneous = np.append(world_coordinates, 1)  # (4,)

# 2. 使用外参矩阵将世界坐标转换为相机坐标
R = camera_extrinsics[:3, :3]
T = camera_extrinsics[:3, 3]
R_inv = R.T
T_inv = -R_inv @ T
camera_extrinsics_inv = np.eye(4)
camera_extrinsics_inv[:3, :3] = R_inv
camera_extrinsics_inv[:3, 3] = T_inv
camera_coordinates_homogeneous = camera_extrinsics_inv @ world_coordinates_homogeneous  # (4,)
camera_coordinates = camera_coordinates_homogeneous[:3]  # 取前 3 个值 (X, Y, Z)

# 3. 使用内参矩阵将相机坐标转换为像素坐标
u = (camera_intrinsics[0, 0] * camera_coordinates[0] / camera_coordinates[2]) + camera_intrinsics[0, 2]
v = (camera_intrinsics[1, 1] * camera_coordinates[1] / camera_coordinates[2]) + camera_intrinsics[1, 2]

# 打印像素坐标
result_pixel_coords = (u, v)
print(f"Ori Pixel Coordinates: {pixel_coords},Result Pixel Coordinates: {result_pixel_coords}")


