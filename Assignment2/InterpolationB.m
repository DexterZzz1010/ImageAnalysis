clc
clear
close all

% Define original data points
x = 1:8;
f = [3 4 7 4 3 5 6 12];

% Define interpolation points
xi = 1:0.1:8; % Interpolate between original data points

%% Linear Interpolation Function 1: Linear Interpolation

g1 = @(x) (1 - abs(x)) .* (abs(x) <= 1); % Linear interpolation weights
Fi1 = zeros(size(xi));
fi1 = zeros(size(xi));
for j = 1:length(xi)
    for i = 1 : 8
        fi1(j)= g1(xi(j)-i).*f(i);
        Fi1(j)=Fi1(j) + fi1(j);
    end
end


%% Linear Interpolation Function 2: Defined Interpolation
g2 = @(x) cos(pi/2 * abs(x)) .* (abs(x) <= 1)-(pi/2) *(abs(x)^3 - 5*abs(x)^2 + 8*abs(x)-4).* (abs(x) <= 2 && abs(x) > 1); % Cubic interpolation weights
Fi2 = zeros(size(xi));
fi2 = zeros(size(xi));
for j = 1:length(xi)
    for i = 1 : 8
        fi2(j)= g2(xi(j)-i).*f(i);
        Fi2(j)=Fi2(j) + fi2(j);
    end
end

%% Determine whether F2 is differentiable or not 
F2_derivative = diff(Fi2);

if ~any(isnan(F2_derivative))
    disp('F2(x) is differentiable');
else
    disp('F2(x) is not differentiable');
end

%% Plot original data and results of different interpolation methods
figure;
plot(x, f, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Original Data');
hold on;
plot(xi, Fi1, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Linear Interpolation');
plot(xi, Fi2, 'm-', 'LineWidth', 1.5, 'DisplayName', 'Defined Interpolation');
xlabel('Position (x)');
ylabel('Value (f(x)');
title('Comparison of Different Interpolation Methods');
legend('Location', 'Best');
grid on;
hold off;
