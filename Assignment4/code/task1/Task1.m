clear all;
close all;

% Load the image
image = imread('michelangelo_colorshift.jpg');

% Convert the image to double precision
grayWorldImage = im2double(image);

%% Gray World
meanR = mean(mean(grayWorldImage(:,:,1)));
meanG = mean(mean(grayWorldImage(:,:,2)));
meanB = mean(mean(grayWorldImage(:,:,3)));

meanGray = (meanR + meanG + meanB) / 3;

grayWorldImage(:,:,1) = grayWorldImage(:,:,1) * meanGray / meanR;
grayWorldImage(:,:,2) = grayWorldImage(:,:,2) * meanGray / meanG;
grayWorldImage(:,:,3) = grayWorldImage(:,:,3) * meanGray / meanB;

%% White World Assumption
whiteWorldImage = im2double(image);
maxR = max(max(whiteWorldImage(:, :, 1)));
maxG = max(max(whiteWorldImage(:, :, 2)));
maxB = max(max(whiteWorldImage(:, :, 3)));

maxWhite = max([maxR, maxG, maxB]);

whiteWorldImage(:, :, 1) = whiteWorldImage(:, :, 1) * maxWhite / maxR;
whiteWorldImage(:, :, 2) = whiteWorldImage(:, :, 2) * maxWhite / maxG;
whiteWorldImage(:, :, 3) = whiteWorldImage(:, :, 3) * maxWhite / maxB;

%% Display the corrected images
figure;
subplot(1, 3, 1);
imshow(image);
title('Original figure');
subplot(1, 3, 2);
imshow(grayWorldImage);
title('Gray World Assumption');
subplot(1, 3, 3);
imshow(whiteWorldImage);
title('White World Assumption');


%% Load the correct white-balanced image
correctImage = imread('michelangelo_correct.jpg');
correctImage = im2double(correctImage);
% correctImage = double(correctImage);

% Load the three output images
outputImage = imread('result.jpg');
outputImage = im2double(outputImage);
% outputImage = double(outputImage);

% Calculate PSNR
[psnr1, ~] = psnr(grayWorldImage, correctImage);
[psnr2, ~] = psnr(whiteWorldImage, correctImage);
[psnr3, ~] = psnr(outputImage, correctImage);

% Calculate SSIM
ssim1 = ssim(grayWorldImage, correctImage);
ssim2 = ssim(whiteWorldImage, correctImage);
ssim3 = ssim(outputImage, correctImage);

% Calculate LIP-error using the computeFLIP function
addpath('flip-matlab_0/matlab') % Update with the correct path
err1 = computeFLIP(correctImage, grayWorldImage);
err2 = computeFLIP(correctImage, whiteWorldImage);
err3 = computeFLIP(correctImage, outputImage);

% Display the results
fprintf('PSNR:\n');
fprintf('Gray World Image: %.2f\n', psnr1);
fprintf('White World Image: %.2f\n', psnr2);
fprintf('WB Image 3: %.2f\n\n', psnr3);

fprintf('SSIM:\n');
fprintf('Gray World Image: %.4f\n', ssim1);
fprintf('White World Image: %.4f\n', ssim2);
fprintf('WB Image 3: %.4f\n\n', ssim3);

fprintf('LIP-error:\n');
fprintf('Gray World Image: %.2f\n', err1);
fprintf('White World Image: %.2f\n', err2);
fprintf('WB Image 3: %.2f\n', err3);