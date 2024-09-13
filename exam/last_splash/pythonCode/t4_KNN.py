# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/22 20:30
@Auth ： Dexter ZHANG
@File ：t4_KNN.py
@IDE ：PyCharm
"""

# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/21 18:40
@Auth ： Dexter ZHANG
@File ：task4.py
@IDE ：PyCharm
"""

#!/usr/bin/env python3
import cv2
import cv2 as cv
import numpy as np
from PIL import Image
import numpy as np


def Decompose(img):
    img_data = np.array(img)
    # 计算像素数和特征数
    rows, cols, channels = img_data.shape
    num_pixels = rows * cols

    # 将图像数据重塑为(N, 3)形式
    X = np.reshape(img_data, (num_pixels, channels))

    from sklearn.cluster import KMeans

    # 将图像数据聚类为k个颜色
    k = 16
    kmeans = KMeans(n_clusters=k).fit(X)
    labels = kmeans.predict(X)
    colors = kmeans.cluster_centers_.astype(int)

    # 使用聚类中心替换每个像素的颜色
    new_X = colors[labels]
    new_X = new_X.reshape((rows, cols, channels))

    # 将新图像保存到磁盘
    new_img = Image.fromarray(np.uint8(new_X))
    new_img.save("output.jpg")
    return new_img

img_file = r'coin4.jpg' #图像的名称
img = cv2.imread(img_file)

height, width = img.shape[:2]

# 指定新的宽度和高度（原来的1/4）
new_width = width // 4
new_height = height // 4

# 使用resize函数来缩小图像
img = cv2.resize(img, (new_width, new_height))

img = Decompose(img)


# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息
#
# # gay_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)#灰度化
# kernel_size = (5, 5)  # 可根据需要调整内核大小
#
# # 指定高斯内核的标准差
# sigma = 0  # 根据需要调整标准差，0表示自动计算
#
# # 进行高斯滤波
# img = cv2.GaussianBlur(img, kernel_size, sigma)           #中值滤波
#
# gray_img= cv2.cvtColor(img, code=cv2.COLOR_BGRA2GRAY)
# hsv_img = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变
#
# # lower = np.array([0, 0, 100])
# # upper = np.array([100, 200, 255])
#
# lower = np.array([0, 20, 80])
# upper = np.array([200, 120, 170])
#
#
# mask = cv2.inRange(hsv_img, lower, upper)
#
#
# kernel_size = (7, 7)
# # kernel_size = (20, 20)
#
# # 创建一个椭圆形内核
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
#
# # 对图像执行闭操作
# closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#
# res = cv2.bitwise_and(gray_img, gray_img, mask=mask)
#
# edges = cv.Canny(res, 30, 150)
# cv2.imshow('closed_image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# #cv.HoughCircles内部调用cv.Sobel() 可直接输入灰度图像
#
# circles = cv2.HoughCircles(
# 	                     res,                    #输入图像（可直接输入灰度图像）
# 	                     cv2.HOUGH_GRADIENT,       #3元素输出向量（横纵坐标和半径）
# 	                     1,                       #累加器具有与输入图像相同的分辨率
# 	                     40,                      #检测到的圆的中心之间的最小距离（太小的圆抛弃）
# 	                     param1=50,
# 	                     param2=65,
# 	                     minRadius=40,             #最小圆半径
# 	                     maxRadius=110)             #最大圆半径
#
# # coin 0-3
# # circles = cv2.HoughCircles(
# # 	                     res,                    #输入图像（可直接输入灰度图像）
# # 	                     cv2.HOUGH_GRADIENT,       #3元素输出向量（横纵坐标和半径）
# # 	                     1,                       #累加器具有与输入图像相同的分辨率
# # 	                     40,                      #检测到的圆的中心之间的最小距离（太小的圆抛弃）
# # 	                     param1=25,
# # 	                     param2=50,
# # 	                     minRadius=25,             #最小圆半径
# # 	                     maxRadius=100)             #最大圆半径
#
# # coin2.jpg
# # circles = cv2.HoughCircles(
# # 	                     gray_img,                    #输入图像（可直接输入灰度图像）
# # 	                     cv2.HOUGH_GRADIENT,       #3元素输出向量（横纵坐标和半径）
# # 	                     1,                       #累加器具有与输入图像相同的分辨率
# # 	                     50,                      #检测到的圆的中心之间的最小距离（太小的圆抛弃）
# # 	                     param1=40,
# # 	                     param2=80,
# # 	                     minRadius=50,             #最小圆半径
# # 	                     maxRadius=100)             #最大圆半径
#
# # circles = cv2.HoughCircles(
# # 	                     gray_img,                    #输入图像（可直接输入灰度图像）
# # 	                     cv2.HOUGH_GRADIENT,       #3元素输出向量（横纵坐标和半径）
# # 	                     1,                       #累加器具有与输入图像相同的分辨率
# # 	                     100,                      #检测到的圆的中心之间的最小距离（太小的圆抛弃）
# # 	                     param1=100,
# # 	                     param2=100,
# # 	                     minRadius=200,             #最小圆半径
# # 	                     maxRadius=450)             #最大圆半径
#
# if circles is None:
# 	print("None")
# else:
# 	circles = np.uint16(np.around(circles))
# 	print(circles)                #打印圆位置信息 #横坐标纵坐标半径
# 	for i in circles[0, :]:
# 		cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
# 		cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#
# # 使用resize函数来缩小图像
# # img = cv2.resize(img, (new_width, new_height))
# cv2.imshow('circle', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()