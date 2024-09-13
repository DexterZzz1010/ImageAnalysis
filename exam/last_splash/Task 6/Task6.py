import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取两张图片并转换为灰度图像
image1 = cv2.imread('cat_train.png')
image2 = cv2.imread('cat_test1.png')

# 使用高斯滤波对图像进行平滑处理
sigma = 3  # 高斯滤波的标准差
FImage1 = cv2.GaussianBlur(image1, (0, 0), sigma)
FImage2 = cv2.GaussianBlur(image2, (0, 0), sigma)
grayImage1 = cv2.cvtColor(FImage1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(FImage2, cv2.COLOR_BGR2GRAY)

# 已知点A的坐标
xA = 184
yA = 144

# 创建SIFT对象
sift = cv2.SIFT_create()

# 在image1上检测SIFT特征点并提取描述符
keypoints1, features1 = sift.detectAndCompute(grayImage1, None)
N = 500  # 选择前500个最强特征点
keypoints1 = keypoints1[:N]
features1 = features1[:N]

# 在image2上检测SIFT特征点并提取描述符
keypoints2, features2 = sift.detectAndCompute(grayImage2, None)
M = 500  # 选择前500个最强特征点
keypoints2 = keypoints2[:M]
features2 = features2[:M]

# 创建FLANN匹配器
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 匹配image1和image2的特征点
matches = flann.knnMatch(features1, features2, k=2)

# 进行比值测试，筛选出好的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 找到与已知点A匹配的点B
matchedPointB = keypoints2[good_matches[0].trainIdx]
print(matchedPointB.pt)
print(matchedPointB.pt[0])

# 可视化匹配的特征点在原图像上
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(cv2.cvtColor(image1_with_keypoints, cv2.COLOR_BGR2RGB))
axs[0].plot(xA, yA, 'ro', markersize=5, linewidth=2)
axs[0].set_title('Image 1 with Point A')

axs[1].imshow(cv2.cvtColor(image2_with_keypoints, cv2.COLOR_BGR2RGB))
axs[1].plot(matchedPointB.pt[0], matchedPointB.pt[1], 'go', markersize=5, linewidth=1)
axs[1].set_title('Image 2 with Matched Points')

plt.show()