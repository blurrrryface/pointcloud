import os
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 标签映射
label_mapping = {
    0: (0, 255, 0),  # one
    1: (0, 0, 255),  # sparse
    2: (255, 0, 0)   # dense
}

def create_folders(mask_image_folder, result_image_folder):
    os.makedirs(mask_image_folder, exist_ok=True)
    os.makedirs(result_image_folder, exist_ok=True)

def load_model(model_path):
    return YOLO(model_path)

def reconstruct_image(image_size, masks, classes):
    # 创建一个和图片原始大小相同的黑色图像
    reconstructed_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    for mask, cls in zip(masks, classes):
        color = label_mapping.get(cls, [0, 0, 0])
        mask = mask.astype(bool)

        # Resize mask to match image size
        mask_image = Image.fromarray(mask).resize(image_size, Image.Resampling.NEAREST)
        mask_resized = np.array(mask_image).astype(bool)

        reconstructed_image[mask_resized] = color

    return reconstructed_image

def process_images(model, image_paths, mask_image_folder, result_image_folder):
    images = [Image.open(image_path) for image_path in image_paths]
    results = model.predict(images)

    for image_path, result in zip(image_paths, results):
        # 提取检测框和掩码信息
        masks = result.masks.data.cpu().numpy()  # 每个掩码的 numpy 数组
        classes = result.boxes.cls.cpu().numpy()  # 每个掩码对应的类别

        # 重建图像并保存到指定文件夹
        image_size = images[0].size
        mask_image_name = os.path.basename(image_path).replace('.jpg', '_mask.png')
        reconstructed_image = reconstruct_image(image_size, masks, classes)
        Image.fromarray(reconstructed_image).save(os.path.join(mask_image_folder, mask_image_name))

        # 显示并保存预测结果到指定文件夹
        im_array = result.plot()                        # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])       # RGB PIL image
        result_image_name = os.path.basename(image_path).replace('.jpg', '_result.jpg')
        im.save(os.path.join(result_image_folder, result_image_name))  # save image

def traverse_and_process(model, input_folder, mask_image_folder, result_image_folder, batch_size=8):
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for image_name in files:
            if image_name.endswith('.jpg'):
                image_path = os.path.join(root, image_name)
                image_paths.append(image_path)

                if len(image_paths) == batch_size:
                    process_images(model, image_paths, mask_image_folder, result_image_folder)
                    image_paths = []

    # 处理剩余的图像
    if image_paths:
        process_images(model, image_paths, mask_image_folder, result_image_folder)

# 主函数
def main():
    # 定义路径
    mask_image_folder = './mask_images'
    result_image_folder = './result_images'
    input_folder = './ultralytics/D2C'
    model_path = './runs/segment/train6/weights/best.pt'

    # 创建文件夹
    create_folders(mask_image_folder, result_image_folder)

    # 加载模型
    model = load_model(model_path)

    # 遍历文件夹并处理图像
    traverse_and_process(model, input_folder, mask_image_folder, result_image_folder)

if __name__ == '__main__':
    main()