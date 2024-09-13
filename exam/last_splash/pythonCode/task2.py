# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/21 20:10
@Auth ： Dexter ZHANG
@File ：task2.py
@IDE ：PyCharm
"""

import cv2
import numpy as np
import os

def readImage(img_file_path):

    binary_img = None

    img = cv2.imread(img_file_path,0)

    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    binary_img = img

    return binary_img

def blockwork(img,coordinate):
    size = CELL_SIZE

    h = size*(coordinate[0]+1)

    w = size*(coordinate[1]+1)

    h0= size*coordinate[0]

    w0= size*coordinate[1]

    block = img[h0:h,w0:w]

    up = bool(block[0,int(size/2)]) *1000

    down = bool(block[int(size-1),int(size/2)])*100

    left = bool(block[int(size/2),0]) *10

    right = bool(block[int(size/2),int(size-1)])*1

    edge = up+down+left+right

    return edge, block

def solveMaze(original_binary_img, initial_point, final_point, no_cells_height, no_cells_width):

    edgearray = []

    for i in range (no_cells_height):

        edgearray.append([])

        for j in range(no_cells_width):

            sz = [i,j]

            edge, block = blockwork(img, sz)

            edgearray[i].append(edge)

            edge= edgearray

#The solvemaze method is continued here...

shortestPath = []

img = original_binary_img

sp = []

rec = [0]

p = 0

sp.append(list(initial_point))

while True:

    h,w = sp[p][0],sp[p][1]

    #h stands for height and w stands for width

    if sp[-1]==list(final_point):

        break

    if edge[h][w] > 0:

        rec.append(len(sp))

    if edge[h][w]>999:

        #If this edge is open upwards

        edge[h][w] = edge[h][w]-1000

        h = h-1

        sp.append([h,w])

        edge[h][w] =edge[h][w]-100

        p = p+1

        continue

    if edge[h][w]>99:

        #If the edge is open downward

        edge[h][w] =edge[h][w]-100

        h = h+1

        sp.append([h,w])

        edge[h][w] =edge[h][w]-1000

        p=p+1

        continue

    if edge[h][w]>9:

        #If the edge is open left

        edge[h][w] = edge[h][w]-10

        w = w-1

        sp.append([h,w])

        edge[h][w] = edge[h][w]-1

        p = p+1

        continue

    if edge[h][w]==1:

        #If the edge is open right

        edge[h][w] = edge[h][w]-1

        w = w+1

        sp.append([h,w])

        edge[h][w] = edge[h][w]-10

        p=p+1

        continue

    else:

        #Removing the coordinates that are closed or don't show any path

        sp.pop()

        rec.pop()

        p = rec[-1]

for i in sp:

    shortestPath.append(tuple(i))

return shortestPath

def pathHighlight(img, ip, fp, path):

    size = CELL_SIZE

    for coordinate in path:

        h = CELL_SIZE*(coordinate[0]+1)

        w = CELL_SIZE*(coordinate[1]+1)

        h0= CELL_SIZE*coordinate[0]

        w0= CELL_SIZE*coordinate[1]

        img[h0:h,w0:w] = img[h0:h,w0:w]-50

    return img