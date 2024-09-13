clc
clear
close all

x = linspace(-3, 3, 1000);
g = zeros(size(x));

% Define g(x) according to the piecewise definition
g(abs(x) <= 1) = cos(pi/2 * x(abs(x) <= 1));
g(1 < abs(x) & ...
    abs(x) <= 2) = -pi/2 * (abs(x(1 < abs(x) & ...
    abs(x) <= 2)).^3 - 5 * abs(x(1 < abs(x) & ...
    abs(x) <= 2)).^2 + 8 * abs(x(1 < abs(x) & ...
    abs(x) <= 2)) - 4);

% Plot g(x)
plot(x, g);
title('g(x)');
xlabel('x');
ylabel('g(x)');
grid on;
