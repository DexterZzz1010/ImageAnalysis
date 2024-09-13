function net = class_train(X, Y)
% function net = trainSimpleCNN(X, Y)

% 假设每个图像的大小为19x19
nx = 28; 
ny = 28;

% 将输入数据重新格式化
X = reshape(X, ny, nx, 1, []);

% 将标签转换为分类类型
Y = categorical(Y);

% 定义CNN结构
layers = [
    imageInputLayer([ny nx 1])
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% 指定训练选项
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 100, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% 训练网络
net = trainNetwork(X, Y, layers, options);
end