import cv2
import numpy as np

# 读取图像
image = cv2.imread('ultralytics/D2C/1226481/color/000000_color_01226479_1280x720.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# 边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 霍夫圆变换
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                           param1=50, param2=30, minRadius=55, maxRadius=150)

# 确保至少检测到一个圆
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # 遍历所有检测到的圆
    for (x, y, r) in circles:
        # 绘制圆的外边缘
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # 绘制圆心
        cv2.circle(image, (x, y), 2, (0, 128, 255), 3)

# 显示结果图像
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()