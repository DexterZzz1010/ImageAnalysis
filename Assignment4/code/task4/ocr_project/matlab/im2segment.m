function S = im2segment(img)
DISPLAY = 0;
img = uint8(img);
% figure;
% imshow(img);

% img = imbilatfilt(img);
% se = strel('disk', 7);
% img = imtophat(img, se);


% % 定义自定义卷积核
% custom_kernel = -[1/9 1/9 1/9; 1/9 -1 1/9; 1/9 1/9 1/9];
% 
% % 使用imfilter函数进行卷积
% img = imfilter(img, custom_kernel);
% 指定高斯滤波的标准差
sigma = 0.46; % 标准差值越大，平滑效果越明显

% 对图像进行高斯滤波
img = imgaussfilt(img, sigma);
threshold = 0.13; % Set the threshold of image binarize, default 0.23
binary_image = imbinarize(img, threshold); % binarize
% imshow(binary_image);

% binary_image = medfilt2(binary_image, [3, 3]);
% threshold = 0.9;
% binary_image = imbinarize(binary_image, threshold); % binarize
% imshow(binary_image);


% % % 执行闭运算
% se = strel('disk', 1); % 创建一个半径为2的圆形结构元素
% binary_image = imclose(binary_image, se); % 执行闭运算
[pixelCoordinates, labeled_Image, numLabels] = custom_bwlabel(binary_image, 24);

% 如果元素个数大于5
if numLabels > 5
    % 计算每个向量的维数
    dimensions = cellfun(@numel, pixelCoordinates);
    
    % 确定需要去掉的向量个数
    numToRemove = numLabels - 5;
    
    % 从维数最小的向量开始去掉
    [~, sortedIndices] = sort(dimensions);
    removeIndices = sortedIndices(1:numToRemove);
    
    % 保留剩余的向量
    pixelCoordinates(removeIndices) = [];
end
% 
% % 使用bwlabel将二值图像中的连通区域进行标记
% labeledImage = bwlabel(binary_image, 8);
% 
% disp(labeledImage);
% 
% % 使用bwconncomp获取连通区域的属性
% cc = bwconncomp(binary_image, 8);
% connectedComponents = cc.PixelIdxList; % 获取每个连通区域的像素索引
% 
% % 判断cell中的元素个数
% numElements = numel(connectedComponents);
% 
% % 如果元素个数大于5
% if numElements > 5
%     % 计算每个向量的维数
%     dimensions = cellfun(@numel, connectedComponents);
%     
%     % 确定需要去掉的向量个数
%     numToRemove = numElements - 5;
%     
%     % 从维数最小的向量开始去掉
%     [~, sortedIndices] = sort(dimensions);
%     removeIndices = sortedIndices(1:numToRemove);
%     
%     % 保留剩余的向量
%     connectedComponents(removeIndices) = [];
% end

% % 显示连通区域的标记图像
% figure(2);
% imshow(label2rgb(labeledImage, 'hsv', 'k', 'shuffle'));
% title('Sigmentation');


% % 遍历每个连通区域
% for i = 1:numel(connectedComponents)
%     % 创建一个与原始图像大小相同的全黑图像
%     output_image = zeros(size(img, 1), size(img, 2));
%     
%     % 获取当前连通区域的像素索引
%     pixelIdxList = connectedComponents{1,i};
%     
%     % 将当前连通区域的像素设置为白色
%     output_image(pixelIdxList) = 1;
%     S{i} = output_image;
% 
%     % 显示结果图像
%     if DISPLAY == 1
%         figure;
%         imshow(output_image);
%         title('Individual Sigmentation');
%     end
%     
%     
%     
%     % 显示当前连通区域的图像
% %     figure;
% %     imshow(regionImage);
% end

% 显示连通区域的标记图像
% figure(2);
% imshow(label2rgb(labeled_Image, 'hsv', 'k', 'shuffle'));
% title('Sigmentation');


for label = 1:5
    regionImage = zeros(size(img));
    % 获取当前连通区域的元素坐标集合
    coordinates = pixelCoordinates{label};
    
    % 将当前连通区域的像素设置为前景值
    for i = 1:size(coordinates, 1)
        row = coordinates(i, 1);
        col = coordinates(i, 2);
        regionImage(row, col) = 1;
    end

    S{label} = regionImage;
end
    
end



