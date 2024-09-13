clc;
clear;
close all;
load('imdata_pineapple.mat')
% 步骤1: 加载两张图片
image1 = imread('pineapple2.jpg'); % 加载第一张图片
image2 = imread('pineapple_t1.jpg'); % 加载第二张图片
% 

load('imdata_pineapple.mat'); 

xy2 = u2;
xy1 = u1;

rgb1 = impixel(image1, xy1(:, 1), xy1(:, 2));
rgb2 = impixel(image2, xy2(:, 1), xy2(:, 2));

% 将 RGB 值分别提取出来
R1 = rgb1(:, 1);
G1 = rgb1(:, 2);
B1 = rgb1(:, 3);

R2 = rgb2(:, 1);
G2 = rgb2(:, 2);
B2 = rgb2(:, 3);

% 数据预处理
rgb1 = double(rgb1);
rgb2 = double(rgb2);

% RANSAC 参数
numIterations = 10000; % 迭代次数
inlierThreshold = 10; % 内点阈值
maxInliers_R = 0;
maxInliers_G = 0;
maxInliers_B = 0;
best_kc_R = 0;
best_bc_R = 0;
best_kc_G = 0;
best_bc_G = 0;
best_kc_B = 0;
best_bc_B = 0;

for i = 1:numIterations
    % 随机选择两个数据点
    indices = randperm(size(rgb1, 1), 2);
    sample_rgb1 = rgb1(indices, :);
    sample_rgb2 = rgb2(indices, :);
    
    % 计算 kc 和 bc
    kc_R = sum(sample_rgb2(:, 1) .* sample_rgb1(:, 1)) / sum(sample_rgb1(:, 1) .* sample_rgb1(:, 1));
    bc_R = mean(sample_rgb2(:, 1) - kc_R * sample_rgb1(:, 1));
    
    kc_G = sum(sample_rgb2(:, 2) .* sample_rgb1(:, 2)) / sum(sample_rgb1(:, 2) .* sample_rgb1(:, 2));
    bc_G = mean(sample_rgb2(:, 2) - kc_G * sample_rgb1(:, 2));
    
    kc_B = sum(sample_rgb2(:, 3) .* sample_rgb1(:, 3)) / sum(sample_rgb1(:, 3) .* sample_rgb1(:, 3));
    bc_B = mean(sample_rgb2(:, 3) - kc_B * sample_rgb1(:, 3));
    
    % 计算内点数目
    residuals = abs(rgb2 - (kc_R * rgb1 + bc_R));
    inliers = sum(residuals(:, 1) < inlierThreshold & residuals(:, 2) < inlierThreshold & residuals(:, 3) < inlierThreshold);
    
    % 更新最佳模型
    if inliers > maxInliers_R
        maxInliers_R = inliers;
        best_kc_R = kc_R;
        best_bc_R = bc_R;
    end
end

for i = 1:numIterations
    % 随机选择两个数据点
    indices = randperm(size(rgb1, 1), 2);
    sample_rgb1 = rgb1(indices, :);
    sample_rgb2 = rgb2(indices, :);
    
    kc_G = sum(sample_rgb2(:, 2) .* sample_rgb1(:, 2)) / sum(sample_rgb1(:, 2) .* sample_rgb1(:, 2));
    bc_G = mean(sample_rgb2(:, 2) - kc_G * sample_rgb1(:, 2));

    
    % 计算内点数目
    residuals = abs(rgb2 - (kc_G * rgb1 + bc_G));
    inliers = sum(residuals(:, 1) < inlierThreshold & residuals(:, 2) < inlierThreshold & residuals(:, 3) < inlierThreshold);
    
    % 更新最佳模型
    if inliers > maxInliers_G
        maxInliers_G = inliers;
        best_kc_G = kc_G;
        best_bc_G = bc_G;
    end
end

for i = 1:numIterations
    % 随机选择两个数据点
    indices = randperm(size(rgb1, 1), 2);
    sample_rgb1 = rgb1(indices, :);
    sample_rgb2 = rgb2(indices, :);
    
    % 计算 kc 和 bc

    kc_B = sum(sample_rgb2(:, 3) .* sample_rgb1(:, 3)) / sum(sample_rgb1(:, 3) .* sample_rgb1(:, 3));
    bc_B = mean(sample_rgb2(:, 3) - kc_B * sample_rgb1(:, 3));
    
    % 计算内点数目
    residuals = abs(rgb2 - (kc_B * rgb1 + bc_B));
    inliers = sum(residuals(:, 1) < inlierThreshold & residuals(:, 2) < inlierThreshold & residuals(:, 3) < inlierThreshold);
    
    % 更新最佳模型
    if inliers > maxInliers_B
        maxInliers_B = inliers;
        best_kc_B = kc_B;
        best_bc_B = bc_B;
    end
end

% 应用最佳模型
% 还原image2的色彩

restored_R = (image2(:,:,1) - best_bc_R) / best_kc_R;
restored_G = (image2(:,:,2) - best_bc_G) / best_kc_G;
restored_B = (image2(:,:,3) - best_bc_B) / best_kc_B;

% restored_R = (image2(:,:,1)* best_kc_R + best_bc_R) ;
% restored_G = (image2(:,:,2)* best_kc_G + best_bc_G) ;
% restored_B = (image2(:,:,3)* best_kc_B + best_bc_B) ;

% 将还原后的RGB值转换为8位整数
restored_rgb2 = uint8(cat(3, restored_R, restored_G, restored_B));




figure;
subplot(1, 2, 1);
imshow(image1);
title('ori Image 1');
subplot(1, 2, 2);
imshow(restored_rgb2);
title(' 2');

