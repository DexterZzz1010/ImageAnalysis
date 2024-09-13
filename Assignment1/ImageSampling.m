% Define the resolution of the discrete image
num_pixels = 5;
% Define the number of gray levels
num_gray_levels = 32;

% Create a meshgrid for the x and y values
[x, y] = meshgrid(linspace(0, 1, num_pixels), linspace(1, 0, num_pixels));

% Calculate the intensity values for each pixel using the given function
intensity = x .* (1 - y);

% Quantize the intensity values to 32 gray levels
quantized_intensity = round(intensity * (num_gray_levels - 1));

disp(quantized_intensity);

% Create a colormap for the 32 gray levels
colormap(gray(num_gray_levels));

% Display the discrete image
imagesc(quantized_intensity);
axis off;
title('Quantized Image');
colorbar;
