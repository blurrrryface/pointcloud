import cv2
import numpy as np
import open3d as o3d
import os

# 读取图像
color_image_path = 'E:/ultralytics-main/ultralytics/D2C/1231925/color/000000_color_01231923_1280x720.jpg'
mask_image_path = 'E:/ultralytics-main/mask_images/000000_color_01231923_1280x720_mask.png'
depth_image_path = 'E:/ultralytics-main/ultralytics/D2C/1231925/depth/000000_depth_01231925_1280x720.raw'

color_image = cv2.imread(color_image_path)  # 彩色图像
if color_image is None:
    raise ValueError("Error: 彩色图像未能加载。检查文件路径。")

mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # 掩码图像
if mask_image is None:
    raise ValueError("Error: 掩码图像未能加载。检查文件路径。")

# 自定义读取 .raw 文件
width = 1280
height = 720
depth_image = np.fromfile(depth_image_path, dtype=np.uint16).reshape((height, width))
if depth_image is None:
    raise ValueError("Error: 深度图像未能加载。检查文件路径。")

# 相机内参（示例值，需要根据实际相机参数进行设置）
fx = 689.66  # 焦距 x
fy = 689.66  # 焦距 y
cx = 644.11  # 光心 x
cy = 363.21  # 光心 y
scaling_factor = 5000.0  # 深度图像的缩放因子，将深度值从毫米转换为米

# 获取图像尺寸
height, width = depth_image.shape

# 生成点云
points = []
colors = []
for v in range(height):
    for u in range(width):
        if mask_image[v, u] > 0:  # 仅处理掩码区域
            z = depth_image[v, u] / scaling_factor
            if z == 0:  # 跳过深度为0的点
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(color_image[v, u] / 255.0)  # 记录对应的颜色并归一化

# 转换为NumPy数组
points = np.array(points)
colors = np.array(colors)

# 创建旋转矩阵进行围绕X轴旋转180°
angle = np.pi  # 180度对应的弧度值
rotation_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(angle), -np.sin(angle)],
    [0, np.sin(angle), np.cos(angle)]
])

# 应用旋转矩阵
points = points.dot(rotation_matrix.T)

# 创建 open3d 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 保存点云到文件
output_dir = 'E:/ultralytics-main/point_images'
os.makedirs(output_dir, exist_ok=True)
base_name = os.path.basename(color_image_path)
output_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.ply')
o3d.io.write_point_cloud(output_path, pcd)
print(f"点云已保存到 {output_path}")

# 可视化点云
o3d.visualization.draw_geometries([pcd], window_name='彩色点云', width=1280, height=720)
