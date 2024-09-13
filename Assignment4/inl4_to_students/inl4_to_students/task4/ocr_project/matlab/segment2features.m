function features = segment2features(I)
%% 计算最小外接矩形
stats = regionprops(I, 'BoundingBox');
bounding_box = stats.BoundingBox;

% 步骤1：获取边界框的位置和大小
x = bounding_box(1);
y = bounding_box(2);
width = ceil(bounding_box(3));
height = ceil(bounding_box(4));

% 步骤2：创建一个新的大图像（例如，原始图像的两倍大小）
newWidth = width + 4;  % 扩充两圈黑色像素
newHeight = height + 4;
newImage = zeros(newHeight, newWidth, 'uint8');  % 初始化为黑色

% 步骤3：将原始图像复制到新图像中
newImage(3:end-2, 3:end-2) = I(y:y+height-1, x:x+width-1);

% 步骤4：缩放新图像为28x28
scaledImage = imresize(newImage, [28, 28]);

flattenedImage = reshape(scaledImage, [],1);

%%

% % % 提取HOG特征
 hog_features = extractHOGFeatures(scaledImage);
 hog_features = hog_features';
%%
features = flattenedImage;




