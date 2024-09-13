function predictedClass = features2class(X, model)
% function Y = predictSimpleCNN(net, X)

% 假设每个图像的大小为19x19
nx = 28; 
ny = 28;

% 将输入数据重新格式化
X = reshape(X, ny, nx, 1, []);

% 预测类别
Y = predict(model, X);

% 获取预测结果中概率最高的类别
[~, predictedClass] = max(Y, [], 2);

% 将类别转换为+1/-1
Y = (predictedClass * 2) - 1;

% 将结果转换为行向量
predictedClass = Y(:)';
end