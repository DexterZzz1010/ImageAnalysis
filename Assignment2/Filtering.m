clc
clear
close all

% 读取图像（这里使用了一个示例图像，请替换为您自己的图像）
image = imread('img.jpg');
image = rgb2gray(image); % 如果图像是彩色的，将其转换为灰度图像

% 定义自定义卷积核
custom_kernel = 1/3*[
    1, 1, 0;
    1, 0, -1;
    0, -1, -1
];

% custom_kernel = 1/25*[1 1 1 1 1;
% 1 1 1 1 1;
% 1 1 1 1 1;
% 1 1 1 1 1;
% 1 1 1 1 1
% ];


% custom_kernel = [
%     0, -1, 0;
%     -1, 5, -1;
%     0, -1, 0
% ];

% custom_kernel = [
%     0, 0, 0;
%     0, 1, 0;
%     0, 0, 0
% ];
% 
% custom_kernel = [
%     1, -2, 1
% ];



% 执行卷积操作
convolved_image = conv2(double(image), custom_kernel, 'same'); % 'same'用于保持输出大小与输入相同

% plot the orignal image and the covlutioned image
subplot(1, 2, 1);
imshow(uint8(image));
title('Original Image');

subplot(1, 2, 2);
imshow(uint8(convolved_image));
title('Convolved Image');

% 如果需要，保存卷积后的图像
%imwrite(uint8(convolved_image), 'convolved_image5.jpg');