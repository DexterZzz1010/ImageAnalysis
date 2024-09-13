clc
clear 
close all
% Load the dataset
load('assignment1bases.mat');

% Initialize variables to store the mean error norms
mean_error_norms = zeros(2, 3); % Rows: test sets, Columns: bases

%% Iterate through the test sets (general and face)
for test_set = 1:2
    % Select the test images from the corresponding stack
    test_images = stacks{test_set};
      
    % Iterate through the three bases
    for basis_idx = 1:3
        % Select the basis for this iteration
        basis = bases{basis_idx};
        
        % Initialize an array to store error norms for individual images
        error_norms = zeros(size(test_images, 3), 1);
        % Iterate through all test images
        for img_idx = 1:400
            % Get the current test image
            img = test_images(:, :, img_idx);
            
            % Project the image onto the basis and calculate the error norm
            [up, r] = projectAndCalculateError(img, basis);
            
            % Store the error norm
            error_norms(img_idx) = r;
        end
       
        % Calculate the mean error norm for this basis and test set
        mean_error_norm = mean(error_norms);
        
        % Store the result in the mean_error_norms matrix
        mean_error_norms(test_set, basis_idx) = mean_error_norm;

    end
end

%% print result
% choose the test image
plotset_idx=2;
plotimg_idx=375;
test_images = stacks{plotset_idx};
plot_img = zeros(19,19,3);
    for basis_idx = 1:3
        basis = bases{basis_idx};
        [plot_up, r] = projectAndCalculateError(test_images(:, :, plotimg_idx), basis);
        plot_img(:,:,basis_idx)=plot_up;
    end

% plot results images
figure;
subplot(1, 4, 1);
imshow(uint8(test_images(:, :, plotimg_idx)));
title('Test Image');


subplot(1, 4, 2);
imshow(uint8(plot_img(:,:,1)));
title('Projection1');

subplot(1, 4, 3);
imshow(uint8(plot_img(:,:,2)));
title('Projection2');

subplot(1, 4, 4);
imshow(uint8(plot_img(:,:,3)));
title('Projection3');

basis1 = abs(bases{1});
min_value = min(basis1(:));
max_value = max(basis1(:));
basis1 = 255 * (basis1 - min_value) / (max_value - min_value);
basis1 = uint8(basis1);

figure;
title('Basis3');
subplot(1, 4, 1);
imshow(basis1(:, :, 1));
subplot(1, 4, 2);
imshow(basis1(:, :, 2));
subplot(1, 4, 3);
imshow(basis1(:, :, 3));
subplot(1, 4, 4);
imshow(basis1(:, :, 4));

% Display or use the mean_error_norms matrix as needed
disp('Mean Error Norms:');
disp(mean_error_norms);