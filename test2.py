import cv2
import numpy as np

# 读取图像
image = cv2.imread('ultralytics/D2C/1226481/color/000000_color_01226479_1280x720.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 直方图均衡化
gray = cv2.equalizeHist(gray)

# 使用高斯模糊以减少噪声
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 使用形态学操作来去除噪声和连接边缘
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=1)

# 检测轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Detected Mushrooms', image)
cv2.waitKey(0)
cv2.destroyAllWindows()