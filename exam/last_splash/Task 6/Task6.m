clc;
clear;
close all;
% 读取两张图片并转换为灰度图像
image1 = imread('cat_train.png');
image2 = imread('cat_test1.png');

% 使用高斯滤波对图像进行平滑处理
sigma = 3; % 高斯滤波的标准差
FImage1 = imgaussfilt(image1, sigma);
FImage2 = imgaussfilt(image2, sigma);
grayImage1 = rgb2gray(FImage1);
grayImage2 = rgb2gray(FImage2);


% 已知点A的坐标
xA = 184;
yA = 144;

% 在image1上检测SURF特征点并提取描述符
points1 = detectSURFFeatures(grayImage1);
N = 500; % 选择前500个最强特征点
points1 = selectStrongest(points1, N);
[features1, validPoints1] = extractFeatures(grayImage1, points1);

% 在image2上检测SURF特征点并提取描述符
points2 = detectSURFFeatures(grayImage2);
M =  500; % 选择前500个最强特征点
points2 = selectStrongest(points2, M);
[features2, validPoints2] = extractFeatures(grayImage2, points2);

% 匹配image1和image2的特征点
indexPairs = matchFeatures(features1, features2);
matchedPoints1 = validPoints1(indexPairs(:, 1));
matchedPoints2 = validPoints2(indexPairs(:, 2));

% 找到与已知点A匹配的点B
distances = sqrt(sum((matchedPoints1.Location - [xA, yA]).^2, 2));
[minDistance, minIndex] = min(distances);
matchedPointB = matchedPoints2(minIndex);
disp(matchedPointB)
disp(matchedPointB.Location(1))

% 可视化匹配的特征点在原图像上
figure;
subplot(1, 2, 1);
imshow(image1);
hold on;
plot(xA, yA, 'ro', 'MarkerSize', 5, 'LineWidth', 2);
title('Image 1 with Point A');

subplot(1, 2, 2);
imshow(image2);
hold on;
plot(254, 159, 'go', 'MarkerSize', 5, 'LineWidth', 1);
% plot(matchedPointB.Location(1), matchedPointB.Location(2), 'go', 'MarkerSize', 5, 'LineWidth', 1);
title('Image 2 with Matched Points');