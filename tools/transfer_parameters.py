import numpy as np
from scipy.spatial.transform import Rotation as R

# 提取平移向量和欧拉角
# translation_vector = [1.0532510813527192, 0.11058064969153436, 0.9816117632129159]
# euler_angles = [-2.7246307295250696, -0.031247157971078243, 1.617240122129939]
transformation_left = [0.85225669,0.55364429,0.54680531,-2.32181209,-0.02543146, 2.67303257]
transformation_top = [ 0.92639657, -0.00502778,  0.84467367, -2.37464738,  0.04213155,  1.57273205]
transformation = transformation_top
translation_vector = [transformation[0],transformation[1],transformation[2]]
euler_angles = [transformation[3],transformation[4],transformation[5]]

# 将欧拉角转换为旋转矩阵
rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()

# 创建4x4外参矩阵
extrinsic_matrix = np.eye(4)
extrinsic_matrix[:3, :3] = rotation_matrix
extrinsic_matrix[:3, 3] = translation_vector

print("Extrinsic Matrix [R|t]:")
print(extrinsic_matrix)

# 结果对应：
    # top_camera_extrinsics = np.array([
    #                                 [-0.04640444,  0.91275018, -0.40587405,  1.05325108],
    #                                 [ 0.99843405,  0.0550883 ,  0.0097323 ,  0.11058065],
    #                                 [ 0.03124207, -0.40478685, -0.9138772 ,  0.98161176],
    #                                 [ 0.        ,  0.        ,  0.        ,  1.        ]])