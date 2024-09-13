function features = segment2features(I)

% 计算每一列的白色像素数量
whitePixelCount = sum(I, 1);

% 确定左边界和右边界
leftBoundary = find(whitePixelCount > 0, 1, 'first');
rightBoundary = find(whitePixelCount > 0, 1, 'last');

% 截取白色部分
whiteRegion = I(:, leftBoundary:rightBoundary, :);
desiredSize = [28, 28];  % 目标大小
currentSize = size(whiteRegion);  % 当前大小

% 创建一个全黑的图像，大小为目标大小
InterstR = uint8(zeros(desiredSize(1), desiredSize(2), size(whiteRegion, 3)));

% 将白色区域复制到黑色区域的适当位置
InterstR(1:currentSize(1), 1:currentSize(2), :) = whiteRegion;

% 定义卷积核
kernel = [1, 0, -1; 2, 0, -2; 1, 0, -1];  % 示例卷积核

% 进行卷积操作
convolvedImage = conv2(double(InterstR), kernel, 'same');

% 将 convolvedImage 转换为列向量
features = convolvedImage(:);