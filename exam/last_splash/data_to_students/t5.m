clc;
clear;
close all;
load('imdata_pineapple.mat')
% 步骤1: 加载两张图片
image1 = imread('pineapple2.jpg'); % 加载第一张图片(gt)
image2 = imread('pineapple_t2.jpg'); % 加载第二张图片
% 
% image1 = im2double(image1);
% image2 = im2double(image2);

xy1 = u2;
xy2 = u1;

x=xy2(:, 1);
y=xy2(:, 2);

rgb1 = impixel(image1, xy1(:, 1), xy1(:, 2)); %(gt)
rgb2 = impixel(image2, xy2(:, 1), xy2(:, 2));


% 将 RGB 值分别提取出来

% gt
R1 = rgb1(:, 1);
G1 = rgb1(:, 2);
B1 = rgb1(:, 3);

R2 = rgb2(:, 1);
G2 = rgb2(:, 2);
B2 = rgb2(:, 3);

% 数据预处理
rgb1 = double(rgb1);
rgb2 = double(rgb2);


%% R

% 将数据整合成一个表格
T_R = table(R1, R2, x, y);

% 使用多元线性回归模型拟合数据
lm_R = fitlm(T_R, 'R1 ~ R2 + x + y ');

% 获取回归系数
Coefficients = lm_R.Coefficients.Estimate;

% Coefficients 包含了 A, B, C 和 D 的估计值
A_R = double(Coefficients(2));
B_R = double(Coefficients(3));
C_R = double(Coefficients(4));
D_R = double(Coefficients(1));

%%
% 将数据整合成一个表格
T_G = table(G1, G2, x, y);
% 使用多元线性回归模型拟合数据
lm_G = fitlm(T_G, 'G1 ~ G2 + x + y ');

% 获取回归系数
Coefficients = lm_G.Coefficients.Estimate;

% Coefficients 包含了 A, B, C 和 D 的估计值
A_G = double(Coefficients(2));
B_G = double(Coefficients(3));
C_G = double(Coefficients(4));
D_G = double(Coefficients(1));

%%
% 将数据整合成一个表格
T_B = table(B1, B2, x, y);
% 使用多元线性回归模型拟合数据
lm_B = fitlm(T_B, 'B1 ~ B2 + x + y');

% 获取回归系数
Coefficients = lm_B.Coefficients.Estimate;

% Coefficients 包含了 A, B, C 和 D 的估计值
A_B = double(Coefficients(2));
B_B = double(Coefficients(3));
C_B = double(Coefficients(4));
D_B = double(Coefficients(1));

%%
% 应用最佳模型
% 还原image2的色彩
image_width = 3024;
image_height = 4032;

X = zeros(image_height, image_width);
Y = zeros(image_height, image_width);

% 填充 x 和 y 矩阵
for i = 1:image_height
    for j = 1:image_width
        X(i, j) = double(j);  % 横坐标
        Y(i, j) = double(i);  % 纵坐标
    end
end

img2_R = double(image2(:,:,1));
R_1=B_R * X ;
R_2=C_R * Y;


% restored_R = (double(image2(:,:,1)) - B_R * X - C_R * Y - D_R) / A_R;
% restored_G = (double(image2(:,:,2)) - B_G * X - C_G * Y - D_G) / A_G;
% restored_B = (double(image2(:,:,3)) - B_B * X - C_B * Y - D_B) / A_B;

restored_R = (double(image2(:,:,1))*A_R + B_R * X + C_R * Y + D_R)  ;
restored_G = (double(image2(:,:,2))*A_G + B_G * X + C_G * Y + D_G) ;
restored_B = (double(image2(:,:,3))*A_B - B_B * X + C_B * Y + D_B);

% 将还原后的RGB值转换为8位整数
restored_rgb2 = uint8(cat(3, restored_R, restored_G, restored_B));

% 显示还原结果
figure;
subplot(1, 2, 1);
imshow(image1);
title('原始图像 1');
subplot(1, 2, 2);
imshow(restored_rgb2);
title('还原后的图像 2');


