function features = segment2features(I)
run('D:\MATLAB\vlfeat-0.9.21\toolbox/vl_setup');% 计算周长
stats = regionprops(I, 'Perimeter', 'Area');
perimeter = stats.Perimeter;
area = stats.Area;
compactness = perimeter^2 / (4*pi*area);

%%
% 计算面积
stats = regionprops(I, 'Area');
area = stats.Area;

ac_ratio = area/compactness ; 

% % 计算凸包面积比
% stats = regionprops(I, 'Area', 'ConvexHull');
% area = stats.Area;
% convex_hull_area = polyarea(stats.ConvexHull(:, 1), stats.ConvexHull(:, 2));
% convex_hull_ratio = area / convex_hull_area;



histogram_features = sum(I, 2);

%%
radius_range = [6, 15]; % 圆的半径范围
sensitivity = 0.9; % 灵敏度，根据需要调整
edge_threshold = 0.1; % 边缘阈值，根据需要调整
[centers, radii] = imfindcircles(I, radius_range, 'Sensitivity', sensitivity, 'EdgeThreshold', edge_threshold);
num_circles = length(centers);

%%
skeleton = bwmorph(I, 'skel', Inf);
skeleton_length = sum(skeleton(:));

lbp_features = extractLBPFeatures(I);
lbp_features = lbp_features';

cell_size = [8, 8];
block_size = [2, 2];
num_bins = 9;

%% 计算最小外接矩形
stats = regionprops(I, 'BoundingBox');
bounding_box = stats.BoundingBox;
bounding_box_area = bounding_box(3) * bounding_box(4);
bounding_box_ratio = bounding_box(3) / bounding_box(4);


% 将bounding_box分成3x3的近似区域
num_rows = 3;
num_cols = 3;

% 初始化一个9x1的向量来存储黑白像素数量比
pixel_ratio_vector_3 = zeros(num_rows * num_cols, 1);

% 计算每个小区域的黑白像素数量比
index = 1;
for row = 1:num_rows
    for col = 1:num_cols
        % 计算每个小区域的宽度和高度
        small_region_width = bounding_box(3) / num_cols;
        small_region_height = bounding_box(4) / num_rows;
        
        % 计算当前小区域的边界框坐标
        x = round(bounding_box(1) + (col - 1) * small_region_width);
        y = round(bounding_box(2) + (row - 1) * small_region_height);
        width = round(small_region_width);
        height = round(small_region_height);
        
        % 提取当前小区域
        small_region = I(y:y+height-1, x:x+width-1);
        
        % 计算黑白像素数量比
        black_pixels = sum(small_region(:) == 0);
        white_pixels = sum(small_region(:) == 1);
        pixel_ratio =  white_pixels / black_pixels ;
        
        % 存储在向量中
        pixel_ratio_vector_3(index) = pixel_ratio;

        % 获取白色像素的位置
        [y_pos, x_pos] = find(small_region == 1);
        white_pixel_positions{index} = [x_pos, y_pos];
        
        index = index + 1;
    end
end
%%
% % 将bounding_box分成3x3的近似区域
% num_rows = 4;
% num_cols = 4;
% 
% % 初始化一个9x1的向量来存储黑白像素数量比
% pixel_ratio_vector_4 = zeros(num_rows * num_cols, 1);
% 
% % 计算每个小区域的黑白像素数量比
% index = 1;
% for row = 1:num_rows
%     for col = 1:num_cols
%         % 计算每个小区域的宽度和高度
%         small_region_width = bounding_box(3) / num_cols;
%         small_region_height = bounding_box(4) / num_rows;
%         
%         % 计算当前小区域的边界框坐标
%         x = round(bounding_box(1) + (col - 1) * small_region_width);
%         y = round(bounding_box(2) + (row - 1) * small_region_height);
%         width = round(small_region_width);
%         height = round(small_region_height);
%         
%         % 提取当前小区域
%         small_region = I(y:y+height-1, x:x+width-1);
%         
%         % 计算黑白像素数量比
%         black_pixels = sum(small_region(:) == 0);
%         white_pixels = sum(small_region(:) == 1);
%         pixel_ratio =  white_pixels / black_pixels ;
%         
%         % 存储在向量中
%         pixel_ratio_vector_4(index) = pixel_ratio;
% 
%         % 获取白色像素的位置
%         [y_pos, x_pos] = find(small_region == 1);
%         white_pixel_positions{index} = [x_pos, y_pos];
%         
%         index = index + 1;
%     end
% end


% 现在，pixel_ratio_vector 中包含了每个小区域的黑白像素数量比
%%
% % 对每个小区域内的白色像素进行线性回归，并记录斜率和截距
% slopes = zeros(num_rows * num_cols, 1);
% intercepts = zeros(num_rows * num_cols, 1);
% 
% for i = 1:num_rows * num_cols
%     white_pixels = white_pixel_positions{i};
%     if isempty(white_pixels)
%         % 如果没有白色像素点，将斜率和截距设置为0
%         slopes(i) = 0;
%         intercepts(i) = 0;
%     else
%         x = white_pixels(:, 1);
%         y = white_pixels(:, 2);
% 
%         % 执行线性回归
%         coefficients = polyfit(x, y, 1);
% 
%         % 提取斜率和截距
%         slope = coefficients(1);
%         intercept = coefficients(2);
% 
%         % 存储斜率和截距
%         slopes(i) = slope;
%         intercepts(i) = intercept;
%     end
% end


% for i = 1:length(stats)
%     % 获取当前bounding box的坐标
%     bbox = round(stats(i).BoundingBox);
%     x = bbox(1);
%     y = bbox(2);
%     width = bbox(3);
%     height = bbox(4);
%     
%     % 提取bounding box中的区域
%     bbox_region = I(y:y+height-1, x:x+width-1);
%     hog_features = extractHOGFeatures(bbox_region, 'CellSize', cell_size, 'BlockSize', block_size, 'NumBins', num_bins);
%     hog_features = hog_features';
% end

%%
% 假设 I 是你的黑白图像

% % 使用Harris角点检测
% corners = detectHarrisFeatures(I);
% 
% % 计算检测到的角点数量
% num_corners = size(corners, 1);
%%
% % 提取感兴趣区域
% x = round(stats.BoundingBox(1));
% y = round(stats.BoundingBox(2));
% width = round(stats.BoundingBox(3)+2);
% height = round(stats.BoundingBox(4)+2);
% roi_img = imcrop(I, [x, y, width, height]);
% 
% 
% % 使用VLFeat的vl_sift函数进行SIFT特征提取
% [frames, descriptors] = vl_sift(single(roi_img));

% % 限制提取的特征点数量为5
% num_features_to_extract = 5;
% 
% % 选择最强的五个特征点
% [~, strongest_indices] = maxk(frames(3, :), num_features_to_extract);
% 
% % 提取最强的特征点的SIFT描述子
% selected_descriptors = descriptors(:, strongest_indices);
% 
% % 将SIFT描述子存储为5*1的向量
% selected_descriptor_vector = reshape(selected_descriptors, [], 1);


%%
% % 使用Hough变换检测线条
% [H, T, R] = hough(I);
% P = houghpeaks(H, 4); % 根据需要选择适当数量的峰值
% 
% % 提取检测到的线条
% lines = houghlines(I, T, R, P);
% 
% % 初始化交叉点计数
% crossing_count = 0;
% 
% % 遍历检测到的线条，查找X型交叉点
% for i = 1:length(lines)
%     for j = i+1:length(lines)
%         theta1 = lines(i).theta;
%         theta2 = lines(j).theta;
%         
%         % 设置角度差阈值，根据需要调整
%         angle_threshold = 10; % 度数
%         
%         % 检查两条线是否接近垂直方向
%         if abs(theta1 - theta2) < angle_threshold
%             % 获取两条线的交点
%             x1 = lines(i).point1;
%             x2 = lines(i).point2;
%             x3 = lines(j).point1;
%             x4 = lines(j).point2;
%             
%             % 计算交点
%             intersection_point = lineIntersect(x1, x2, x3, x4);
%             
%             % 如果交点有效（不为空），则增加交叉点计数
%             if ~isempty(intersection_point)
%                 crossing_count = crossing_count + 1;
%             end
%         end
%     end
% end
% 


%%

% % % 提取HOG特征
%  hog_features = extractHOGFeatures(I, 'CellSize', cell_size, 'BlockSize', block_size, 'NumBins', num_bins);
% %  hog_features = HOG(I);
%  hog_features = hog_features';
%%
%features = [perimeter;compactness;area;area_features;convex_hull_ratio;histogram_features];
%features = [perimeter;compactness;area;convex_hull_ratio;histogram_features];
% features = [perimeter;compactness;area;skeleton_length;num_circles;convex_hull_ratio;histogram_features];
%features = [num_circles;histogram_features;convex_hull_ratio;hog_features];
% features = [num_circles;histogram_features;hog_features];
%features = hog_features;
% features = [ac_ratio;bounding_box_ratio;num_circles;pixel_ratio_vector];
% features = pixel_ratio_vector;
%features = [num_circles;pixel_ratio_vector_3;pixel_ratio_vector_4];
 features = [num_circles;pixel_ratio_vector_3];
%  features = crossing_count;
% features= descriptors;





