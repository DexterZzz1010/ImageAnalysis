%% Matlab stub for task 2 in assignment 4 in Image analysis

clc;
clear;
close all;

load heart_data % load data

figure;
imshow(im)


% 提取两个类别的像素强度
chamber_pixels = chamber_values;
background_pixels = background_values;

% 计算两个类别的像素强度的均值和标准差
chamber_mean = mean(chamber_pixels);
chamber_std = std(chamber_pixels);

background_mean = mean(background_pixels);
background_std = std(background_pixels);

% 输出结果
fprintf('Chamber class: mean = %f, std = %f\n', chamber_mean, chamber_std);
fprintf('Background class: mean = %f, std = %f\n', background_mean, background_std);

M = 96; % height of image, change this!
N = 96; % width of image, change this!

n = M*N; % Number of image pixels

% create neighbour structure

Neighbours = edges4connected(M,N); % use 4-neighbours (or 8-neighbours with edges8connected)

i=Neighbours(:,1); 
j=Neighbours(:,2);
A = sparse(i,j,1,n,n); % create sparse matrix of connections between pixels 


% We can make A into a graph, and show it (test this for example for M = 5, N = 6 to
% see. For the full image it's not easy to see structure)

Ag = graph(A);
figure;
plot(Ag);

% Choose weights:
% weights = abs(im(i) - im(j));


sigma = 0.5; % 你需要设置这个值
diff = im(i) - im(j);
weights = exp(-diff.^2 / (2*sigma^2));

% [Gx, Gy] = gradient(double(im));
% gradient_i = sqrt(Gx(i).^2 + Gy(i).^2);
% gradient_j = sqrt(Gx(j).^2 + Gy(j).^2);
% weights = abs(gradient_i - gradient_j);


% Decide how important a short curve length is:
lambda = 0.1;


% A = A+lambda*speye(n);  % set regularization term so  that A_ij = lambda
  
mu1 = normfit(background_values);
mu2 = normfit(chamber_values);

% 计算负对数似然性
pixels = im(:);
neg_log_likelihood_chamber = (pixels - chamber_mean).^2 / (2 * chamber_std^2);
neg_log_likelihood_background = (pixels - background_mean).^2 / (2 * background_std^2);
Tt = sparse((pixels - chamber_mean).^2 / (2 * chamber_std^2)); % 设置源节点的数据项
Ts = sparse((pixels - background_mean).^2 / (2 * background_std^2)); % 设置汇节点的数据项


% Ts = sparse((im(:)-mu1).^2); % set weights to source, according to assignment!
% % Ts = sparse((im(:)-mu1).^2); for standard weighting without statistical weight
% Tt = sparse((im(:)-mu2).^2); % set weights to sink, according to assignment!
% % Tt = sparse((im(:)-mu2).^2); for standard weighting without statistical weight

% create matrix of the full graph, adding source and sink as nodes n+1 and
% n+2 respectively

F = sparse(zeros(n+2,n+2));
F(1:n,1:n) = A; % set regularization weights
F(n+1,1:n) = Ts'; % set data terms 
F(1:n,n+1) = Ts; % set data terms 
F(n+2,1:n) = Tt'; % set data terms 
F(1:n,n+2) = Tt; % set data terms 

% make sure that you understand what the matrix F represents!

Fg = graph(F); % turn F into a graph Fg

% help maxflow % see how Matlab's maxflow function works

[MF,GF,CS,CT] = maxflow(Fg,n+1,n+2); % run maxflow on graph with source node (n+1) and sink node (n+2)

disp(MF) % shows the optmization value (maybe not so interesting)

% CS contains the pixels connected to the source node (including the source
% node n+1 as final entry (CT contains the sink nodes).

% We can construct out segmentation mask using these indices
seg = zeros(M,N);
seg(CS(1:end-1)) = 1; % set source pixels to 1
figure;
imagesc(seg); % show segmentation 







