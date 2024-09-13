
# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/23 14:39
@Auth ： Dexter ZHANG
@File ：sift.py
@IDE ：PyCharm
"""

import cv2
import numpy as np

# 读取两张连续图像
image1 = cv2.imread('cat_train.png')
image2 = cv2.imread('cat_test2.png')

grayImage1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
grayImage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 在两个图像上检测关键点和计算描述符
keypoints1, descriptors1 = sift.detectAndCompute(grayImage1, None)
keypoints2, descriptors2 = sift.detectAndCompute(grayImage2, None)

# 使用FLANN（快速库近似最近邻）匹配器进行特征点匹配
index_params = dict(algorithm=0, trees=10)
search_params = dict(checks=50,crossCheck=True)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 选择好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append(m)

# 提取匹配点的坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 连接匹配的特征点
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

# 使用RANSAC算法估算变换矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 输入图像1上的点坐标
input_points = np.array([[184, 144], [586, 130], [326, 386], [474, 376],[412, 492]],dtype=np.float32).reshape(-1, 1, 2)


# 通过变换矩阵将图像1上的点坐标变换为图像2上的坐标
output_points = cv2.perspectiveTransform(input_points, M)
# output_points[output_points < 0] = 5
output_points = np.abs(output_points)

# 打印坐标值
for i, point in enumerate(output_points):
    x, y = point[0]
    print(f"Point {i + 1} in Image 1: ({input_points[i][0][0]:.2f}, {input_points[i][0][1]:.2f})")
    print(f"Transformed Point in Image 2: ({x:.2f}, {y:.2f})")

# 在图像2上标记坐标
for point in output_points:
    x, y = point[0]
    cv2.circle(image2, (int(x), int(y)), 5, (0, 0, 255), -1)

# 保存标记后的图像
cv2.imwrite('marked_image2.jpg', image2)

# 显示匹配的特征点和标记后的图像
cv2.imshow('Matched Features', matched_image)
cv2.imshow('Marked Image 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
