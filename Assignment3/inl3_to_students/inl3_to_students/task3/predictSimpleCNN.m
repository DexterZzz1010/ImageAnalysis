function Y = predictSimpleCNN(net,X)
% function Y = predictSimpleCNN(net,X)


% reformat input data
% input images assumed to be 19x19
nx = 19; 
ny = 19;
X = reshape(X,ny,nx,1,[]);

% predict classes
Y = predict(net,X);

% transform to +1/-1
Y = round(Y(:,2))*2-1;
% make row vector 
Y = Y(:)';
