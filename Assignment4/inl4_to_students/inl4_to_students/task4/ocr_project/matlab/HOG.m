function average_hog_feature = HOG(large_image)
large_image_height = size(large_image, 1);
large_image_width = size(large_image, 2);

% 每个小图像的尺寸
small_image_height = 28;
small_image_width = 28;

% 初始化一个存储小图像的单元数组
small_images = cell(1, 5);

% 分割大图像成五个小图像
for i = 1:5
    % 计算小图像的位置
    start_row = 1;
    start_col = (i - 1) * small_image_width + 1;
    end_row = small_image_height;
    end_col = i * small_image_width;
    
    % 提取小图像
    small_images{i} = large_image(start_row:end_row, start_col:end_col);
end

% 初始化一个存储HOG特征的单元数组
hog_features = cell(1, 5);

% 计算每个小图像的HOG特征
for i = 1:5
    % 计算HOG特征
    cell_size = [8, 8];
    block_size = [2, 2];
    num_bins = 9;
    hog = extractHOGFeatures(small_images{i}, 'CellSize', cell_size, 'BlockSize', block_size, 'NumBins', num_bins);
    
    % 存储HOG特征
    hog_features{i} = hog;
end

% 计算HOG特征的平均值
average_hog_feature = mean(cat(1, hog_features{:}));
end

