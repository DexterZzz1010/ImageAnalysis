function [pixelCoordinates, labels, numLabels] = custom_bwlabel(image, c)
    [rows, cols] = size(image);
    labels = zeros(rows, cols);  % 创建一个与原图像相同大小的零矩阵
    nextLabel = 1;  % 初始化下一个可用的标签

    % 遍历图像像素
    for i = 1:rows
        for j = 1:cols
            % 如果当前像素为前景且未标记
            if image(i, j) ~= 0 && labels(i, j) == 0
                % 执行深度优先搜索进行连通区域标记
                labels = dfs(image, labels, i, j, nextLabel, c);
                nextLabel = nextLabel + 1;  % 更新下一个可用的标签
            end
        end
    end

    numLabels = nextLabel - 1;  % 计算连通区域的数量
    pixelCoordinates = cell(1, numLabels);

    % 遍历所有像素
for row = 1:rows
    for col = 1:cols
        % 获取当前像素的标记
        label = labels(row, col);
        
        % 如果当前像素属于某个连通组件
        if label > 0
            % 添加当前像素的坐标到相应连通组件的向量中
            pixelCoordinates{label} = [pixelCoordinates{label}; [row, col]];
        end
    end
end

% 创建一个空的cell数组，用于存储每个连通区域的左上角像素坐标
regionPositions = cell(1, numLabels);

% 遍历所有像素，记录每个连通区域的左上角像素坐标
for row = 1:rows
    for col = 1:cols
        label = labels(row, col);
        if label > 0
            % 如果当前像素所属的连通区域还没有记录左上角像素坐标，则记录下来
            if isempty(regionPositions{label})
                regionPositions{label} = [row, col];
            end
        end
    end
end

% 对连通区域进行排序，根据左上角像素的x坐标
[~, sortedIndices] = sort(cellfun(@(x) x(2), regionPositions));
% 创建一个新的cell数组，用于存储排序后的连通区域坐标
sortedPixelCoordinates = cell(1, numLabels);

% 根据排序后的顺序调整连通区域坐标的顺序
for i = 1:numLabels
    label = sortedIndices(i);
    sortedPixelCoordinates{i} = pixelCoordinates{label};
end

% 将排序后的连通区域坐标覆盖原始的pixelCoordinates
pixelCoordinates = sortedPixelCoordinates;
end


function labels = dfs(image, labels, row, col, label, c)
    [rows, cols] = size(image);
    stack = zeros(rows * cols, 2);  % 创建一个堆栈用于实现深度优先搜索
    stackSize = 1;
    stack(1, :) = [row, col];
    labels(row, col) = label;

    while stackSize > 0
        current = stack(stackSize, :);
        stackSize = stackSize - 1;

        % 检查当前像素的上、下、左、右四个邻域(4-connect)
        neighbors_4 = [current(1)-1, current(2);
                     current(1)+1, current(2);
                     current(1), current(2)-1;
                     current(1), current(2)+1];
        % 8-connect
        neighbors_8 = [current(1)-1, current(2);
                     current(1)+1, current(2);
                     current(1), current(2)-1;
                     current(1), current(2)+1;
                     current(1)-1, current(2)-1;
                     current(1)-1, current(2)+1;
                     current(1)+1, current(2)-1;
                     current(1)+1, current(2)+1];
        %
        neighbors_24 = [current(1)-1, current(2)-1;
             current(1)-1, current(2);
             current(1)-1, current(2)+1;
             current(1), current(2)-1;
             current(1), current(2)+1;
             current(1)+1, current(2)-1;
             current(1)+1, current(2);
             current(1)+1, current(2)+1;
             current(1)-2, current(2)-2;
             current(1)-2, current(2)-1;
             current(1)-2, current(2);
             current(1)-2, current(2)+1;
             current(1)-2, current(2)+2;
             current(1)-1, current(2)-2;
             current(1)-1, current(2)+2;
             current(1), current(2)-2;
             current(1), current(2)+2;
             current(1)+1, current(2)-2;
             current(1)+1, current(2)+2;
             current(1)+2, current(2)-2;
             current(1)+2, current(2)-1;
             current(1)+2, current(2);
             current(1)+2, current(2)+1;
             current(1)+2, current(2)+2];

        % 检查当前像素的上、下、左、右、斜对角线和对角线方向的邻域
neighbors_48 = [current(1)-1, current(2)-1;
             current(1)-1, current(2);
             current(1)-1, current(2)+1;
             current(1), current(2)-1;
             current(1), current(2)+1;
             current(1)+1, current(2)-1;
             current(1)+1, current(2);
             current(1)+1, current(2)+1;
             current(1)-2, current(2)-2;
             current(1)-2, current(2)-1;
             current(1)-2, current(2);
             current(1)-2, current(2)+1;
             current(1)-2, current(2)+2;
             current(1)-1, current(2)-2;
             current(1)-1, current(2)+2;
             current(1), current(2)-2;
             current(1), current(2)+2;
             current(1)+1, current(2)-2;
             current(1)+1, current(2)+2;
             current(1)+2, current(2)-2;
             current(1)+2, current(2)-1;
             current(1)+2, current(2);
             current(1)+2, current(2)+1;
             current(1)+2, current(2)+2;
             current(1)-3, current(2)-3;
             current(1)-3, current(2)-2;
             current(1)-3, current(2)-1;
             current(1)-3, current(2);
             current(1)-3, current(2)+1;
             current(1)-3, current(2)+2;
             current(1)-3, current(2)+3;
             current(1)-2, current(2)-3;
             current(1)-2, current(2)+3;
             current(1)-1, current(2)-3;
             current(1)-1, current(2)+3;
             current(1), current(2)-3;
             current(1), current(2)+3;
             current(1)+1, current(2)-3;
             current(1)+1, current(2)+3;
             current(1)+2, current(2)-3;
             current(1)+2, current(2)+3;
             current(1)+3, current(2)-3;
             current(1)+3, current(2)-2;
             current(1)+3, current(2)-1;
             current(1)+3, current(2);
             current(1)+3, current(2)+1;
             current(1)+3, current(2)+2;
             current(1)+3, current(2)+3];
        
        
        if c == 4
            neighbors = neighbors_4;
        elseif c == 8
            neighbors = neighbors_8;
        elseif c == 24
            neighbors = neighbors_24;
        elseif c == 48
            neighbors = neighbors_48;
        end

        for k = 1:size(neighbors, 1)
            neighborRow = neighbors(k, 1);
            neighborCol = neighbors(k, 2);

            % 检查邻域是否在图像范围内，并且像素值不为零并且标签未分配
            if neighborRow >= 1 && neighborRow <= rows && neighborCol >= 1 && neighborCol <= cols && image(neighborRow, neighborCol) ~= 0 && labels(neighborRow, neighborCol) == 0
                stackSize = stackSize + 1;
                stack(stackSize, :) = [neighborRow, neighborCol];
                labels(neighborRow, neighborCol) = label;
            end
        end
    end
end