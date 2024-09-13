clc;
clear;
close all;
load('imdata_pineapple.mat')
% 步骤1: 加载两张图片
image1 = imread('pineapple2.jpg'); % 加载第一张图片(gt)
image2 = imread('pineapple_t1.jpg'); % 加载第二张图片

xy2 = u1;
xy1 = u2;

rgb1 = impixel(image1, xy1(:, 1), xy1(:, 2)); %(gt)
rgb2 = impixel(image2, xy2(:, 1), xy2(:, 2));
% 将 RGB 值分别提取出来
% 数据预处理
rgb1 = double(rgb1);
rgb2 = double(rgb2);
% gt
R1 = rgb1(:, 1);
G1 = rgb1(:, 2);
B1 = rgb1(:, 3);

R2 = rgb2(:, 1);
G2 = rgb2(:, 2);
B2 = rgb2(:, 3);



%% R
% 使用最小二乘法拟合
X = [R2, ones(size(R1))];

% 使用最小二乘法拟合
paramsR = X \ R1;

% 提取拟合结果
A_R = paramsR(1);
B_R = paramsR(2);

%% G
% 使用最小二乘法拟合

X = [G2, ones(size(G1))];

% 使用最小二乘法拟合
paramsG = X \ G1;

% 提取拟合结果
A_G = paramsG(1);
B_G = paramsG(2);


%% B
% 使用最小二乘法拟合

X = [B2, ones(size(B1))];

% 使用最小二乘法拟合
paramsB = X \ B1;

% 提取拟合结果
A_B = paramsB(1);
B_B = paramsB(2);



%%
% 应用最佳模型
% 还原image2的色彩

% restored_R = (image2(:,:,1) - B_R) / A_R;
% restored_G = (image2(:,:,2) - B_G) / A_G;
% restored_B = (image2(:,:,3) - B_B) / A_B;

restored_R = (image2(:,:,1)*A_R + B_R) ;
restored_G = (image2(:,:,2)*A_G + B_G) ;
restored_B = (image2(:,:,3)*A_B + B_B) ;


% 将还原后的RGB值转换为8位整数
restored_rgb2 = uint8(cat(3, restored_R, restored_G, restored_B));

% 显示还原结果
figure;
subplot(1, 2, 1);
imshow(image1);
title('ori Image 1');
subplot(1, 2, 2);
imshow(restored_rgb2);
title(' 2');


