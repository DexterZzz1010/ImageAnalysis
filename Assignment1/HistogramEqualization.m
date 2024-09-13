% Define the range
r = linspace(0, 1, 256);

% Define  pr
pr = 6 * r .* (1 - r);

% Calculate F(r)
F_r = cumtrapz(r, pr);

% Calculate the inverse of the F(r) to get the transformation s = T(r)
F_inverse = interp1(F_r, r, linspace(0, 1, numel(r)));

% Create a figure to visualize the transformation matrix T(r)
figure;
plot(r, F_inverse, 'b', 'LineWidth', 2);
xlabel('Original Intensity (r)');
ylabel('Equalized Intensity (s)');
title('Transformation Function T(r)');
grid on;
