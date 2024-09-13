# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/22 19:54
@Auth ： Dexter ZHANG
@File ：CoinTemplate.py
@IDE ：PyCharm
"""

#!/usr/bin/env python3
import cv2
import cv2 as cv
import numpy as np

original = cv2.imread('cat_test1.png')
img = cv2.imread('cat_test1.png', 0)
template = cv2.imread('rightEar.png', 0)
h, w = template.shape[:2]
ret = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)

#取匹配程度大于80%的坐标
index = np.where(ret > 0.972)

draw_img = original.copy()
for i in zip(*index[::-1]):	#*代表可选参数
    rect = cv2.rectangle(draw_img, i, (i[0]+w, i[1]+h), (0, 0, 255), 1)

output = np.hstack((original, rect))


cv2.imshow('rect', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
