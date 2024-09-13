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
title('Line Fitting with Total Least Squares and RANSAC')

x_fine = [min(xm) - 0.05, max(xm) + 0.05]; % Used when plotting the fitted lines

% Fit a line to these data points with Total Least Squares (TLS)
X_tls = [xm, ones(N, 1)];
[U, S, V] = svd(X_tls);
p_tls = -V(1, 2) / V(1, 1); % Coefficient of the TLS line
plot(x_fine, p_tls * x_fine + V(2, 2), 'g', 'LineWidth', 2); % TLS Line

% Fit a line to these data points using RANSAC and TLS on the inlier set
best_inliers = [];
best_p_ransac = [0, 0];
best_error = Inf;
num_iterations = 1000;
inlier_threshold = 0.1;

for i = 1:num_iterations
    % Randomly select two points
    sample_indices = randsample(N, 2);
    x_sample = xm(sample_indices);
    y_sample = ym(sample_indices);
    
    % Fit a line to the two sampled points using TLS
    X_sample = [x_sample, ones(2, 1)];
    [~, ~, V_sample] = svd(X_sample);
    p_ransac_tls = -V_sample(1, 2) / V_sample(1, 1);
    
    % Compute errors for the RANSAC solution
    errors = abs(p_ransac_tls * xm + V_sample(2, 2) - ym);
    inliers = find(errors < inlier_threshold);
    inlier_count = length(inliers);
    
    % Update the best model if this model has more inliers
    if inlier_count > length(best_inliers)
        best_inliers = inliers;
        best_p_ransac = [p_ransac_tls, V_sample(2, 2)];
        best_error = sum(errors.^2);
    end
end

% Plot the RANSAC + TLS Line
plot(x_fine, best_p_ransac(1) * x_fine + best_p_ransac(2), 'b--', 'LineWidth', 2); % RANSAC + TLS Line

% Compute errors for the TLS line
errors_tls = abs(p_tls * xm + V(2, 2) - ym);
least_square_error_tls = sum(errors_tls.^2);
total_least_square_error_tls = sum(errors_tls.^2 / (1 + p_tls^2));

% Compute errors for the RANSAC + TLS line (on the inlier set)
errors_ransac_tls = abs(best_p_ransac(1) * xm(best_inliers) + best_p_ransac(2) - ym(best_inliers));
least_square_error_ransac_tls = sum(errors_ransac_tls.^2);
total_least_square_error_ransac_tls = sum(errors_ransac_tls.^2 / (1 + best_p_ransac(1)^2));

% Display the computed errors
fprintf('TLS Least Square Error: %.4f\n', least_square_error_tls);
fprintf('TLS Total Least Square Error: %.4f\n', total_least_square_error_tls);
fprintf('RANSAC+TLS Least Square Error: %.4f\n', least_square_error_ransac_tls);
fprintf('RANSAC+TLS Total Least Square Error: %.4f\n', total_least_square_error_ransac_tls);

% Legend
legend('Data Points', 'TLS Line', 'RANSAC+TLS Line');

% Comment on the results in your report.