clc
clear
close all

% Define the basis images
phi1 = 1/2 * [1 0 -1; 1 0 -1; 0 0 0; 0 0 0];
phi2 = 1/3 * [1 1 1; 1 0 1; -1 -1 -1; 0 -1 0];
phi3 = 1/3 * [0 1 0; 1 1 1; 1 0 1; 1 1 1];
phi4 = 1/2 * [0 0 0; 0 0 0; 1 0 -1; 1 0 -1];


% Define the original image f
f = [-2 6 3; 13 7 5; 7 1 8; -3 4 4];

% Verify orthonormality of basis images
% orthonormality = isequal(round(phi1 .* phi2), zeros(size(3))) && ...
%     isequal(round(phi1 .* phi3), zeros(size(3))) && ...
%     isequal(round(phi1 .* phi4), zeros(size(3))) && ...
%     isequal(round(phi2 .* phi3), zeros(size(3))) && ...
%     isequal(round(phi2 .* phi4), zeros(size(3))) && ...
%     isequal(round(phi3 .* phi4), zeros(size(3)));

p=dot(phi3(:), phi4(:));

orthonormality = isequal(norm(phi1,1),1) && isequal(norm(phi2,1),1) && ...
    isequal(norm(phi3,1),1) && isequal(norm(phi4,1),1) &&...
    isequal(dot(phi3(:), phi4(:)), 0) && ...
    isequal(dot(phi1(:), phi2(:)), zeros(size(3))) && ...
    isequal(dot(phi1(:), phi3(:)), zeros(size(3))) && ...
    isequal(dot(phi1(:), phi4(:)), zeros(size(3))) && ...
    isequal(dot(phi2(:), phi3(:)), zeros(size(3))) && ...
    isequal(dot(phi2(:), phi4(:)), zeros(size(3)));


%% pseudo-inverse
% % Stack the basis images into a matrix
% A = [phi1(:), phi2(:), phi3(:), phi4(:)];
% 
% % Calculate the coefficients using the pseudo-inverse
% x = pinv(A) * f(:);
% 
% % Reconstruct the approximate image
% fa = A * x;
% 
% 
% fa_matrix = reshape(fa, 4, 3);

%% 
% Stack the basis images into a matrix
A = [phi1(:), phi2(:), phi3(:), phi4(:)];

% Calculate the coefficients using the
x = A \ f(:);

% Reconstruct the approximate image
fa = A * x;

% Calculate the approximation error
approximation_error = sum(abs(f(:) - fa).^2);
% Reshape fa
fa_matrix = reshape(fa, 4, 3);
%%
% Calculate the element-wise absolute difference
abs_diff = abs(f(:) - fa);
% Calculate the average difference
f_norm = (norm(f(:), 'fro'))^2;
diff_norm=(norm(abs_diff(:), 'fro'))^2;
% relative_diff = norm_difference/ f_norm;
diff = diff_norm/ f_norm;

%%
% Display results
disp('Orthonormality of Basis Images:');
disp(['Are basis images orthonormal ? orthonormality = ', num2str(orthonormality)]);
disp('Coordinates (x1, x2, x3, x4):');
disp(x);
disp('Approximate Image fa:');
disp(fa_matrix);
disp(['Norm Approximation Error: ', num2str(approximation_error)]);
disp(['my diff: ',num2str(diff)]);