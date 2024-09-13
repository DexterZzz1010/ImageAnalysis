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

img_file = r'coin4.jpg' #图像的名称
img = cv2.imread(img_file)

height, width = img.shape[:2]

# 指定新的宽度和高度（原来的1/4）
new_width = width // 4
new_height = height // 4

# 使用resize函数来缩小图像
img = cv2.resize(img, (new_width, new_height))
ori_img=img
#print('图像的大小是:{}'.format(img.shape)) #打印图像的维度信息

# gay_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)#灰度化
kernel_size = (5, 5)  # 可根据需要调整内核大小
kernel_size = 5
# 指定高斯内核的标准差
sigma = 0  # 根据需要调整标准差，0表示自动计算

# 进行高斯滤波
# img = cv2.GaussianBlur(img, kernel_size, sigma)           #中值滤波
img = cv2.medianBlur(img, kernel_size)


gray_img= cv2.cvtColor(img, code=cv2.COLOR_BGRA2GRAY)
hsv_img = cv2.cvtColor(img, code=cv2.COLOR_BGR2HSV)  # 颜色空间的转变

lower = np.array([0, 0, 100])
upper = np.array([100, 200, 255])

# lower = np.array([0, 20, 80])
# upper = np.array([200, 120, 170])


mask = cv2.inRange(hsv_img, lower, upper)


# kernel_size = (5, 5)
kernel_size = (20, 20)

# 创建一个椭圆形内核
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

# 对图像执行闭操作
closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
opened_image = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

res = cv2.bitwise_and(gray_img, gray_img, mask=closed_image)


cv2.imshow('closed_image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()



# circles = cv2.HoughCircles(
# 	                     gray_img,                    #输入图像（可直接输入灰度图像）
# 	                     cv2.HOUGH_GRADIENT,       #3元素输出向量（横纵坐标和半径）
# 	                     1,                       #累加器具有与输入图像相同的分辨率
# 	                     40,                      #检测到的圆的中心之间的最小距离（太小的圆抛弃）
# 	                     param1=25,
# 	                     param2=50,
# 	                     minRadius=25,             #最小圆半径
# 	                     maxRadius=100)             #最大圆半径

circles = cv2.HoughCircles(
	                     gray_img,                    #输入图像（可直接输入灰度图像）
	                     cv2.HOUGH_GRADIENT,       #3元素输出向量（横纵坐标和半径）
	                     1,                       #累加器具有与输入图像相同的分辨率
	                     80,                      #检测到的圆的中心之间的最小距离（太小的圆抛弃）
	                     param1=25,
	                     param2=67,
	                     minRadius=45,             #最小圆半径
	                     maxRadius=100)             #最大圆半径

if circles is None:
	print("None")
else:
	circles = np.uint16(np.around(circles))
	print(circles)                #打印圆位置信息 #横坐标纵坐标半径
	for i in circles[0, :]:
		cv2.circle(ori_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.circle(ori_img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('circle', ori_img)
cv2.waitKey(0)
cv2.destroyAllWindows()