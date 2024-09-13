import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('cat_train.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('cat_test2.png', cv2.IMREAD_GRAYSCALE)

# 创建ORB对象并计算关键点和描述符
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 设置匹配度的阈值
threshold = 3  # 阈值可根据需要进行调整

# 使用BFMatcher进行匹配
matches = bf.match(descriptors1, descriptors2)

# 按照匹配项的距离进行排序
matches = sorted(matches, key=lambda x: x.distance)

# 筛选出匹配度高于阈值的特征点
good_matches = [m for m in matches if m.distance < threshold * matches[0].distance]

# 提取关键点的坐标
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 创建一个新的图像，将两幅图像并排显示
height, width = max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1]
image_combined = np.zeros((height, width, 3), dtype=np.uint8)
image_combined[:image1.shape[0], :image1.shape[1]] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
image_combined[:image2.shape[0], image1.shape[1]:] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

# 绘制连线
for i in range(len(points1)):
    pt1 = (int(points1[i][0][0]), int(points1[i][0][1]))
    pt2 = (int(points2[i][0][0]) + image1.shape[1], int(points2[i][0][1]))
    cv2.line(image_combined, pt1, pt2, (0, 255, 0), 1)

# 显示图像
plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB))
plt.title('Correspondence Visualization'), plt.xticks([]), plt.yticks([])
plt.show()