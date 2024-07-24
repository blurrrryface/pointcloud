import cv2
import numpy as np
import open3d as o3d
import os


def read_image(image_path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(image_path, flags)
    if image is None:
        raise ValueError(f"Error: 图像未能加载。检查文件路径: {image_path}")
    return image


def read_raw_depth_image(depth_image_path, width, height):
    depth_image = np.fromfile(depth_image_path, dtype=np.uint16).reshape((height, width))
    if depth_image is None:
        raise ValueError(f"Error: 深度图像未能加载。检查文件路径: {depth_image_path}")
    return depth_image


def generate_point_cloud(color_image, mask_image, depth_image, intrinsics, scaling_factor):
    fx, fy, cx, cy = intrinsics
    height, width = depth_image.shape
    points, colors, labels = [], [], []

    for v in range(height):
        for u in range(width):
            label = mask_image[v, u]
            if label > 0:  # 仅处理掩码区域
                z = depth_image[v, u] / scaling_factor
                if z == 0:  # 跳过深度为0的点
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)  # 记录对应的颜色并归一化
                labels.append(label)  # 记录对应的标签

    return np.array(points), np.array(colors), np.array(labels)


def rotate_points(points, angle, axis='x'):
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    else:
        raise ValueError("Unsupported rotation axis")

    return points.dot(rotation_matrix.T)


def create_colored_point_cloud(points,colors, labels):

    # 创建点云并设置点和颜色
    pcd_with_labels = o3d.geometry.PointCloud()
    pcd_with_labels.points = o3d.utility.Vector3dVector(points)
    pcd_with_labels.colors = o3d.utility.Vector3dVector(colors)

    return pcd_with_labels


def save_point_cloud(pcd, output_path):
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到 {output_path}")


def save_labels(labels, labels_path):
    np.save(labels_path, labels)
    print(f"标签已保存到 {labels_path}")


def get_image_paths(base_dir, mask_dir):
    color_image_paths = []
    mask_image_paths = []
    depth_image_paths = []

    # 遍历每个子目录
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            color_path = os.path.join(subdir_path, "color")
            depth_path = os.path.join(subdir_path, "depth")

            # 遍历颜色图像文件
            for file in os.listdir(color_path):
                if file.endswith("_1280x720.jpg"):
                    color_image_paths.append(os.path.join(color_path, file))
                    # 构建掩码图像路径
                    mask_filename = file.replace("_1280x720.jpg", "_1280x720_mask.png")
                    mask_image_paths.append(os.path.join(mask_dir, mask_filename))

            for file in os.listdir(depth_path):
                if file.endswith("1280x720.raw"):
                    depth_image_paths.append(os.path.join(depth_path, file))

    return color_image_paths, mask_image_paths, depth_image_paths


def main():
    color_base_dir = './ultralytics/D2C'
    mask_base_dir = './mask_images'
    color_image_paths, mask_image_paths, depth_image_paths = get_image_paths(color_base_dir, mask_base_dir)

    intrinsics = (689.66, 689.66, 644.11, 363.21)
    scaling_factor = 5000.0
    output_dir = './point_images'
    os.makedirs(output_dir, exist_ok=True)

    for color_image_path, mask_image_path, depth_image_path in zip(color_image_paths, mask_image_paths,
                                                                   depth_image_paths):
        color_image = read_image(mask_image_path)
        mask_image = read_image(mask_image_path, cv2.IMREAD_GRAYSCALE)
        depth_image = read_raw_depth_image(depth_image_path, 1280, 720)

        points, colors, labels = generate_point_cloud(color_image, mask_image, depth_image, intrinsics, scaling_factor)
        points = rotate_points(points, np.pi)  # 旋转180度

        pcd_with_labels = create_colored_point_cloud(points, colors, labels)

        base_name = os.path.basename(color_image_path)
        output_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.ply')
        save_point_cloud(pcd_with_labels, output_path)

        labels_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '_labels.npy')
        save_labels(labels, labels_path)

        # o3d.visualization.draw_geometries([pcd_with_labels], window_name='带标签的彩色点云', width=1280, height=720)


if __name__ == "__main__":
    main()
