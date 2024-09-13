clc
% Given data
x = 1:8; % Data points for x
y = [3, 4, 7, 4, 3, 5, 6, 12]; % Corresponding y values

% Values of x for interpolation
xi = 1:0.1:8; % Values of x for interpolation

l_xi = size(xi,2);
yi = zeros(1,l_xi);

% Linear interpolation
l_x = size(x,2);
    for i = 1:l_xi
        for j = 1:l_x-1 
            % 假如需要计算插值公式
            if x(j+1) > xi(i)
                yi(i) = y(j)+(y(j+1)-y(j))/(x(j+1)-x(j))*(xi(i)-x(j));
                break;
            end
            % 假如插值处的数据已经测得了，就直接把值给它，节约计算资源
            if x(j) == xi(i)
                yi(i) = y(j);
                break;
            end
        end
        % 以上没有把最后一个数据点考虑进去,需要加上
        yi(l_xi) = y(l_x);
    end

% Plot the original data points and linear interpolation

plot(x, y, 'o-', 'DisplayName', 'Data Points');
xlim([0, 9]);
ylim([0, 15]);
hold on;
plot(xi, yi, 'r-', 'DisplayName', 'Linear Interpolation');
title('Linear Interpolation of f(x)');
xlabel('x');
ylabel('f(x)');
legend;
grid on;