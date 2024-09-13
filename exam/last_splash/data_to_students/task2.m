clc
clear
close all

% 读取图像
image = imread('pirate1.png');
originalImage = image;
% 将图像转换为灰度图像（如果不已经是灰度图像）
if size(image, 3) == 3
    grayImage = rgb2gray(image);
else
    grayImage = image;
end

% 二值化图像，将黑色背景设置为0，物体设置为1
binaryImage = imbinarize(grayImage,0.35);
invertedImage = imcomplement(binaryImage);

% 使用八连通分割提取连通区域
cc = bwconncomp(invertedImage, 8);
numObjects = cc.NumObjects;

% 为每个连通区域分配不同的颜色
labeledImage = labelmatrix(cc);
labelImage=labeledImage;
rgbImage = label2rgb(labeledImage, 'jet', 'k');

% % 显示结果图像
% imshow(rgbImage);
% title(['Found ', num2str(numObjects), ' objects']);



% 创建当前连通区域的二进制mask
currentRegion =labelImage == 1;
se = strel('disk', 25);  % 调整膨胀的半径
dilatedRegion1 = imdilate(currentRegion, se);

erodedRegion1 = imerode(dilatedRegion1, se);



currentRegion = labelImage == 2;
se = strel('disk', 25);  % 调整膨胀的半径
dilatedRegion2 = imdilate(currentRegion, se);

erodedRegion2 = imerode(dilatedRegion2, se);

mask = abs(erodedRegion1-erodedRegion2);
mask = imcomplement(mask);



% 将mask应用于原始图像
resultImage = originalImage;
resultImage(:,:,1) = uint8(resultImage(:,:,1)) - uint8(mask*255);

% 显示结果图像
imshow(resultImage);

