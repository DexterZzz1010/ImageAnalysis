clc
clear
close all

% Define the input matrix x
x = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0];

% Define Assume matrices
Assume_1 = [1 1 0; 1 0 1; 1 1 0; 1 0 1; 1 1 0];
Assume_2 = [0 1 0; 1 0 1; 1 0 1; 1 0 1; 0 1 0];
Assume_3 = [0 1 0; 1 0 1; 0 1 0; 1 0 1; 0 1 0];

% Concatenate Assume matrices along the third dimension
Assume = cat(3, Assume_1, Assume_2, Assume_3);

% Initialize the Result matrix with zeros
Result = zeros(5, 3, 3);

% Define prior probabilities for each case
priori = [0.35 0.4 0.25];

% Initialize variables
py = 0;
pyx = zeros(1, 3);
prob = zeros(1, 3);

% Iterate through each 5x3x3 submatrix
for i = 1:5
    for j = 1:3
        for k = 1:3
            % Compare the current 5x3 submatrix with matrix Assume
            if Assume(i, j, k) == x(i, j) && Assume(i, j, k) == 1
                Result(i, j, k) = 0.8;
            elseif Assume(i, j, k) == x(i, j) && Assume(i, j, k) == 0
                Result(i, j, k) = 0.7;
            elseif Assume(i, j, k) ~= x(i, j) && Assume(i, j, k) == 1
                Result(i, j, k) = 0.2;
            elseif Assume(i, j, k) ~= x(i, j) && Assume(i, j, k) == 0
                Result(i, j, k) = 0.3;
            end
        end
    end
end

% Calculate pyx and py
for i = 1:3
    pyx(i) = prod(prod(Result(:,:,i)));
    py = py + pyx(i) * priori(i);
end

% Calculate the probability for each case
for i = 1:3
    prob(i) = (pyx(i) * priori(i)) / py;
end

% Display the Compare Result matrix and probabilities
disp("Compare Result:");
disp(Result);
disp("prob:");
disp(prob);