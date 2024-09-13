clc
clear
close all
%% 读取图像
image = imread('coin1.jpg');
% 将图像转换为灰度图像
grayImage = rgb2gray(image);
% enhancedImage = histeq(grayImage);
% 对图像进行模糊处理以减少噪声
 blurred = imgaussfilt(grayImage, 5);
% 应用霍夫圆检测
[centers, radii, metric] = imfindcircles(blurred,[120 320]);
% 保存圆心和半径到变量
save('coin_data.mat', 'centers', 'radii');
% 在图像上绘制检测到的圆
figure;
imshow(image);
hold on;
viscircles(centers, radii,'EdgeColor','b');
% 显示结果图像
title('Hough Circles');

% 记录圆的数量
num_circles = length(centers);
fprintf('检测到的圆的数量：%d\n', num_circles);