function features = segment2features(I)
% 计算周长
stats = regionprops(I, 'Perimeter', 'Area');
perimeter = stats.Perimeter;
area = stats.Area;
compactness = perimeter^2 / (4*pi*area);


% 计算面积
stats = regionprops(I, 'Area');
area = stats.Area;

ac_ratio = area/compactness ; 

% 计算凸包面积比
stats = regionprops(I, 'Area', 'ConvexHull');
area = stats.Area;
convex_hull_area = polyarea(stats.ConvexHull(:, 1), stats.ConvexHull(:, 2));
convex_hull_ratio = area / convex_hull_area;



histogram_features = sum(I, 2);

radius_range = [6, 15]; % 圆的半径范围
sensitivity = 0.9; % 灵敏度，根据需要调整
edge_threshold = 0.1; % 边缘阈值，根据需要调整
[centers, radii] = imfindcircles(I, radius_range, 'Sensitivity', sensitivity, 'EdgeThreshold', edge_threshold);
num_circles = length(centers);


skeleton = bwmorph(I, 'skel', Inf);
skeleton_length = sum(skeleton(:));

lbp_features = extractLBPFeatures(I);
lbp_features = lbp_features';

cell_size = [8, 8];
block_size = [2, 2];
num_bins = 9;

% 计算最小外接矩形
stats = regionprops(I, 'BoundingBox');
bounding_box = stats.BoundingBox;
bounding_box_area = bounding_box(3) * bounding_box(4);
bounding_box_ratio = bounding_box(3) / bounding_box(4);

% 将bounding_box分成3x3的近似区域
num_rows = 5;
num_cols = 5;

% 初始化一个9x1的向量来存储黑白像素数量比
pixel_ratio_vector = zeros(num_rows * num_cols, 1);

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
        black_pixels = 0;
        white_pixels = 0;
        black_pixels = sum(small_region(:) == 0);
        white_pixels = sum(small_region(:) == 1);
        pixel_ratio = white_pixels/black_pixels ;
        
        % 存储在向量中
        pixel_ratio_vector(index) = pixel_ratio;
        index = index + 1;
    end
end

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




% % 提取HOG特征
 hog_features = extractHOGFeatures(I, 'CellSize', cell_size, 'BlockSize', block_size, 'NumBins', num_bins);
%  hog_features = HOG(I);
 hog_features = hog_features';

%features = [perimeter;compactness;area;area_features;convex_hull_ratio;histogram_features];
%features = [perimeter;compactness;area;convex_hull_ratio;histogram_features];
% features = [perimeter;compactness;area;skeleton_length;num_circles;convex_hull_ratio;histogram_features];
 %features = [num_circles;histogram_features;convex_hull_ratio;hog_features];
%    features = [num_circles;histogram_features;convex_hull_ratio];
% features = [num_circles;histogram_features;convex_hull_ratio;hog_features];
features = [pixel_ratio_vector];




