import numpy as np
import open3d as o3d
import cv2

def calculate_geometric_center(points):
    return np.mean(points, axis=0)

def fit_contour_and_find_highest_point(points):
    highest_point = points[np.argmax(points[:, 2])]
    return highest_point

def find_bounding_circle(points):
    points_2d = points[:, :2]  # 只考虑x和y轴
    center, radius = cv2.minEnclosingCircle(points_2d.astype(np.float32))
    return center, radius

def classify_mushrooms(points, labels, threshold):
    unique_labels = np.unique(labels)
    results = {'A': [], 'B': []}
    cluster_centers = []

    for label in unique_labels:
        if label == -1:
            continue  # 忽略噪声点

        cluster_points = points[labels == label]
        geometric_center = calculate_geometric_center(cluster_points)
        cluster_centers.append(geometric_center)

        highest_point = fit_contour_and_find_highest_point(cluster_points)
        center, radius = find_bounding_circle(cluster_points)

        category = 'A' if radius > threshold else 'B'
        results[category].append({
            'geometric_center': geometric_center,
            'highest_point': highest_point,
            'radius': radius
        })

    return results, cluster_centers


# 读取点云文件
point_cloud_path = "point_images/000000_color_01230119_1280x720.ply"
pcd = o3d.io.read_point_cloud(point_cloud_path)

# 获取点和颜色信息
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 打印一些颜色值以检查它们的范围
print("颜色值范围：", colors.min(), colors.max())

# 假设颜色值范围在 [0, 1]，将其缩放到 [0, 255]
colors = (colors * 255).astype(int)

# 标签映射
label_mapping = {
    (0, 255, 0): 0,    # one 绿色
    (0, 0, 255): 1,    # sparse 蓝色
    (255, 0, 0): 2     # dense 红色
}

# 通过颜色反向得到标签
labels = []
for color in colors:
    color_tuple = tuple(color)
    if color_tuple in label_mapping:
        labels.append(label_mapping[color_tuple])
    else:
        print(f"未找到的颜色: {color_tuple}")
        labels.append(-1)  # 使用 -1 表示未找到的标签

labels = np.array(labels)

print(labels)

# 定义的半径阈值
threshold = 4

# 分类蘑菇
results, cluster_centers = classify_mushrooms(points, labels, threshold)

print("分类结果：", results)
print("簇几何中心：", cluster_centers)

# 可视化结果
pcd.colors = o3d.utility.Vector3dVector(colors)

# 创建几何中心点的点云对象
center_points = o3d.geometry.PointCloud()
center_points.points = o3d.utility.Vector3dVector(cluster_centers)
# 将几何中心点的颜色设为黑色
center_colors = np.array([[0, 0, 0] for _ in cluster_centers])
center_points.colors = o3d.utility.Vector3dVector(center_colors)

# 创建一个几何体列表，并将中心点的大小设大一点
geometries = [pcd, center_points]
for center in cluster_centers:
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.001)
    sphere.translate(center)
    sphere.paint_uniform_color([0, 0, 0])  # 将球体颜色设为黑色
    geometries.append(sphere)

# 可视化点云和几何中心点
o3d.visualization.draw_geometries(geometries)