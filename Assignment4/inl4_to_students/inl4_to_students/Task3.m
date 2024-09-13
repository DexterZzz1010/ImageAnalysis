clc
clear all
% Define points in homogeneous coordinates
a1 = [-4; 5; 1];
a2 = [3; -7; 1];
a3 = [-10; 5; 1];
b1 = [3; 2; 1];
b2 = [6; -1; 1];
b3 = [2; -2; 1];

% Define the fundamental matrix F
F = [2, 2, 4;
     3, 3, 6;
    -5, -10, -6];

% Calculate a' * F * b for all combinations
result_b1_a1 = b1' * F * a1;
result_b1_a2 = b1' * F * a2;
result_b1_a3 = b1' * F * a3;
result_b2_a1 = b2' * F * a1;
result_b2_a2 = b2' * F * a2;
result_b2_a3 = b2' * F * a3;
result_b3_a1 = b3' * F * a1;
result_b3_a2 = b3' * F * a2;
result_b3_a3 = b3' * F * a3;
