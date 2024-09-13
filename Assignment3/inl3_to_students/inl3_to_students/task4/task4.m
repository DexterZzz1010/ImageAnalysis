% Task 4: Fit lines to data points, using total least squares and 
% RANSAC + total least squares

% Clear up
clc;
close all;
clearvars;

% Begin by loading data points from linedata.mat
load linedata

N = length(xm); % number of data points

% Plot data
plot(xm, ym, '*'); hold on;
xlabel('x') 
ylabel('y')
title('TLS & RANSAC') % OBS - CHANGE TITLE!
x_fine = [min(xm)-0.05,max(xm)+0.05]; % used when plotting the fitted lines

%% Fit a line to these data points with total least squares
% Here you should write code to obtain the p_ls coefficients (assuming the
% line has the form y = p_ls(1) * x + p_ls(2)).

data = [xm, ym];

N = length(xm);

% % 计算系数矩阵
% A = [sum(xm.^2) - (1/N)*(sum(xm))^2, sum(xm .* ym) - (1/N)*(sum(xm))*(sum(ym));
%      sum(xm .* ym) - (1/N)*(sum(xm))*(sum(ym)), sum(ym.^2) - (1/N)*(sum(ym))^2];
% 
% % 计算特征值和特征向量
% [V, D] = eig(A);
% 
% % % 提取特征值和特征向量
% % lambda1 = D(1, 1);
% % lambda2 = D(2, 2);
% % a1 = V(1, 1);
% % a2 = V(2, 1);
% % b1 = V(1, 2);
% % b2 = V(2, 2);
% % 
% % % 计算c
% % c = -(1/N)*(a1*sum(xm) + b1*sum(ym));
% % 
% % p_ls= -a1/b1;
% % c_ls=-c/b1;
% 
% 
% % 提取特征向量对应的特征值
% eigenvalues = diag(D);
% 
% % 找到最小特征值的索引
% [~, min_index] = min(eigenvalues);
% 
% % 提取对应于最小特征值的特征向量
% min_eigenvector = V(:, min_index);
% 
% % 提取直线的系数 A、B、C
% a = min_eigenvector(1);
% b = min_eigenvector(2);
% c = -(1/N)*(a*sum(xm) + b*sum(ym));
% p_ls= -a/b;
% c_ls=-c/b;

%% Fit a line to these data points with total least squares
% Here you should write code to obtain the p_ls coefficients (assuming the
% line has the form y = p_ls(1) * x + p_ls(2)).

data = [xm, ym];

N = length(xm);


% 计算数据点的均值
mean_x = mean(xm);
mean_y = mean(ym);

% 计算特征值矩阵的各项元素
N = length(xm);
Sxx = sum(xm.^2) - (1/N) * (sum(xm))^2;
Syy = sum(ym.^2) - (1/N) * (sum(ym))^2;
Sxy = sum(xm.*ym) - (1/N) * (sum(xm)) * (sum(ym));

% 构建特征值矩阵
A = [Sxx, Sxy; Sxy, Syy];

% 计算特征值和特征向量
[V, D] = eig(A);

% 提取特征值和特征向量
eigenvalues = diag(D);
eigenvector = V(:, 1); % 选择第一个特征向量

% 归一化特征向量，使得A^2 + B^2 = 1
eigenvector = eigenvector / sqrt(eigenvector(1)^2 + eigenvector(2)^2);

% 提取直线的系数 A、B 和 C
a = eigenvector(1);
b = eigenvector(2);
c = -(a * mean_x + b * mean_y);

% 输出直线的系数
fprintf('拟合的直线方程：%.2f*x + %.2f*y + %.2f = 0\n', a, b, c);

p_ls= -a/b;
c_ls=-c/b;

% Plot the TLS line
plot(x_fine, p_ls * x_fine + c_ls);

% Display the TLS line equation
fprintf('TLS Line Equation: y = %.4fx + %.4f\n', p_ls, c_ls);

%% Fit a line to these data points using RANSAC and total least squares on the inlier set.

numIterations = 1000;
inlierThreshold = 0.1; % Adjust as needed
minInliers = 2; % Minimum number of inliers for a valid model

bestModel = struct('slope', 0, 'intercept', 0);
bestInliers = [];
bestInlierCount = 0;

for iteration = 1:numIterations
    % Randomly sample two data points
    sampleIndices = randperm(N, 2);
    xSample = xm(sampleIndices);
    ySample = ym(sampleIndices);

    % Calculate the line parameters (slope and intercept)
    slope = (ySample(2) - ySample(1)) / (xSample(2) - xSample(1));
    intercept = ySample(1) - slope * xSample(1);

    % Calculate distances to the line for all data points
    distances = abs(slope * xm + intercept - ym);

    % Find inliers (data points within the threshold)
    inliers = find(distances < inlierThreshold);
    inlierCount = length(inliers);

    % Update the best model if more inliers are found
    if inlierCount > bestInlierCount
        bestInlierCount = inlierCount;
        bestInliers = inliers;
        bestModel.slope = slope;
        bestModel.intercept = intercept;
    end
end

% Refit the best model using all inliers
xInliers = xm(bestInliers);
yInliers = ym(bestInliers);
bestModel.slope = (mean(xInliers) * mean(yInliers) - mean(xInliers .* yInliers)) / ...
                  (mean(xInliers)^2 - mean(xInliers.^2));
bestModel.intercept = mean(yInliers) - bestModel.slope * mean(xInliers);

% Plot the best fit line
x_fine = [min(xm) - 0.05, max(xm) + 0.05];
plot(x_fine, bestModel.slope * x_fine + bestModel.intercept, 'k--');

% Display the best-fit line equation
fprintf('RANSAC Line Equation: y = %.4fx + %.4f\n', bestModel.slope, bestModel.intercept);

% Legend --> show which line corresponds to what (if you need to
% re-position the legend, you can modify rect below)
h=legend('data points', 'least-squares','RANSAC');
rect = [0.20, 0.65, 0.25, 0.25];
set(h, 'Position', rect)

% After having plotted both lines, it's time to compute errors for the
% respective lines. Specifically, for each line (the total least squares and the
% RANSAC line), compute the least square error and the total
% least square error. For the RANSAC solution compute errors on inlier set. 
% Note that the error is the sum of the individual
% squared errors for each data point! In total you should get 4 errors. Report these
% in your report, and comment on the results. OBS: Recall the distance formula
% between a point and a line from linear algebra, useful when computing orthogonal
% errors!

% WRITE CODE BELOW TO COMPUTE THE 4 ERRORS
A_ls = -p_ls;
B_ls = 1;

% 计算 RANSAC 线的系数 A 和 B（仅限内点）
A_ransac = -bestModel.slope;
B_ransac = 1;

% 计算 TLS 误差平方和（点到拟合直线上的垂线段的长度的平方和）
% tls_error_sum_ls = sum(abs(A_ls * xm + B_ls * ym + c_ls) ./ sqrt(A_ls^2 + B_ls^2)).^2;
tls_error_sum_ls = sum(((p_ls * xm + c_ls - ym).^2) / (1+p_ls^2));

% 计算 RANSAC TLS 误差平方和（仅限内点）
% tls_error_sum_ransac = sum(abs(A_ransac * xInliers + B_ransac * yInliers + bestModel.intercept) ./ sqrt(A_ransac^2 + B_ransac^2)).^2;
tls_error_sum_ransac = sum(((bestModel.slope * xInliers + bestModel.intercept - yInliers).^2) / (1+bestModel.slope^2));

% 计算总最小二乘法线的预测值
y_ls = p_ls * xm + c_ls;

% 计算 RANSAC 线的预测值（仅限内点）
y_ransac = bestModel.slope * xInliers + bestModel.intercept;

% 计算垂直误差（最小二乘法线）
vertical_errors_ls = ym - y_ls;

% 计算垂直误差（RANSAC 线，仅限内点）
vertical_errors_ransac = yInliers - y_ransac;

% 计算误差平方和
ls_vertical_error_sum = sum(vertical_errors_ls.^2);

ransac_vertical_error_sum = sum(vertical_errors_ransac.^2);

% 输出四个误差
fprintf('LS Vertical Error Sum: %.2f\n', ls_vertical_error_sum);
fprintf('LS Total Error Sum: %.2f\n', tls_error_sum_ls);
fprintf('RANSAC Vertical Error Sum (Inliers Only): %.2f\n', ransac_vertical_error_sum);
fprintf('RANSAC Total Error Sum (Inliers Only): %.2f\n', tls_error_sum_ransac);