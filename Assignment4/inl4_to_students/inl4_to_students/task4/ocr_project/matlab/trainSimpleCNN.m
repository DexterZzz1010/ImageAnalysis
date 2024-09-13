function net = trainSimpleCNN(X,Y)
% function net = trainSimpleCNN(X,Y)


% reformat input data
% input images assumed to be 19x19
nx = 28; 
ny = 28;
X = reshape(X,ny,nx,1,[]);

% classes should be of type categorical
Y = categorical(Y);

% define cnn-structure
layers = [
    imageInputLayer([ny nx 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


% specify training options

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

% train network

net = trainNetwork(X,Y,layers,options);


