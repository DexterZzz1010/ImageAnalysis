clc
clear
close all
warning off

% 读取图像
img = imread('coin1.jpg');

blurred = imgaussfilt(img, 5);
grayImage = rgb2gray(blurred);

% % 显示原始图像
% imshow(img);
% title('原始图像');

% 转换为HSV颜色空间
hsv_img = rgb2hsv(blurred);

% 提取H通道
h_channel = hsv_img(:,:,1);
s_channel = hsv_img(:,:,2);
v_channel = hsv_img(:,:,3);

% 定义色彩阈值范围
h_min = 0; % 适应您的要求
h_max = 0.2; % 适应您的要求

s_min = 0; % 适应您的要求
s_max = 0.5; % 适应您的要求

v_min = 0.7; % 适应您的要求
v_max = 1.0; % 适应您的要求

% 创建色彩阈值掩码
color_mask = (h_channel >= h_min) & (h_channel <= h_max)&...
    (s_channel >= s_min) & (s_channel <= s_max)&...
    (v_channel >= v_min) & (v_channel <= v_max);



% 提取mask
mask = color_mask;

SE = strel('disk', 3); % 这里使用了一个半径为5的圆形SE

% 执行闭运算（Closing）
closed_image = imclose(mask, SE);

% 执行开运算（Opening）
opened_image = imopen(mask, SE);

% 显示色彩阈值掩码
figure;
imshow(closed_image);
title('色彩阈值掩码');

% 使用Canny边缘检测
edge_mask = edge(closed_image, 'Canny');
% 显示色彩阈值掩码
figure;
imshow(edge_mask);
title('edge');


% 霍夫圆检测
% 根据您的图像和应用的参数进行调整
[centers, radii, metric] = imfindcircles(grayImage, [100 320]);

save('coin_data.mat', 'centers', 'radii');
% 在图像上绘制检测到的圆
figure;
imshow(img);
hold on;
viscircles(centers, radii,'EdgeColor','b');
% 显示结果图像
title('Hough Circles');

% 记录圆的数量
num_circles = length(centers);
fprintf('检测到的圆的数量：%d\n', num_circles);
