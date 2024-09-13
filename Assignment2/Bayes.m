clc
clear
close all

% Define matrix O
O = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 1 0 0];

% Define Assume matrices
Assume_1 = [1 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0];
Assume_2 = [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0];
Assume_3 = [0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 1 0];
Assume_4 = [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 1];

% Concatenate Assume matrices along the third dimension
Assume = cat(3, Assume_1, Assume_2, Assume_3, Assume_4);

% Initialize the Result matrix with zeros
Result = zeros(4, 4, 4);

% Define prior probabilities for each case
priori = [0.3 0.2 0.2 0.3];

% Initialize variables
py = 0;
pyx = zeros(1, 4);
prob = zeros(1, 4);

% Iterate through each 4x4x4 submatrix
for i = 1:4
    for j = 1:4
        for k = 1:4
            % Compare the current 4x4 submatrix with matrix O
            % If the elements are the same, set the corresponding result to 0.8; otherwise, set it to 0.2
            Result(i, j, k) = (Assume(i, j, k) == O(i, j)) * 0.8 + (Assume(i, j, k) ~= O(i, j)) * 0.2;
        end
    end
end

% Calculate pyx and py
for i = 1:4
    pyx(i) = prod(prod(Result(:, :, i)));
    py = py + pyx(i) * priori(i);
end

% Calculate the probability for each case
for i = 1:4
    prob(i) = (pyx(i) * priori(i)) / py;
end

% Display the Result matrix and probabilities
disp("Compare Result:");
disp(Result);
disp("prob:");
disp(prob);