function predictedClass = predictSimpleCNN(X,net)
% function Y = predictSimpleCNN(net,X)


% reformat input data
% input images assumed to be 19x19
nx = 28; 
ny = 28;
X = reshape(X,ny,nx,1,[]);

% predict classes
predictions = predict(net,X);

[maxValue, predictedClass] = max(predictions);
end