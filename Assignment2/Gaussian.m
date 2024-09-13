clc
clear 
close all

%% Define class parameters & test measurements
class_params = [
    struct('mean', 0.4, 'variance', 0.01),   % Class 1
    struct('mean', 0.32, 'variance', 0.05),  % Class 2
    struct('mean', 0.55, 'variance', 0.2)   % Class 3
];
 
test_measurements = [
     0.4003; 0.3988; 0.3998; 0.3997; 0.4010; 0.3995; 0.3991;
    0.2554; 0.3139; 0.2627; 0.3802;0.3287; 0.3160; 0.2924;
    0.5632; 0.7687; 0.0524; 0.7586;0.4243; 0.5005; 0.6769
];

% Initialize counter for correct classifications
correct_classifications = 0;
class_probabilities = zeros(size(test_measurements, 1), numel(class_params));

%% Loop through each test measurement
for i = 1:size(test_measurements, 1)
    test_measurement = test_measurements(i, :);   
    for p = 1:length(test_measurement)
        for j = 1:numel(class_params)
            params = class_params(j);
            mean = params.mean;
            variance = params.variance;
            % Calculate likelihood using normal distribution
            likelihood = normpdf(test_measurement(p), mean, variance);
            class_probabilities(i,j) = prod(likelihood);
        end
    end  
end

%% Predict label
[~, predictions] = max(class_probabilities , [], 2); 
% Display the number of correctly classified test measurements
correct_count = sum(predictions == [1; 1; 1; 1; 1; 1;1;2; 2; 2; 2; 2; 2;2;3;3;3;3;3;3;3]);
predictions= reshape(predictions, 7, 3)';
disp('Probabilities : ')
disp(class_probabilities);
disp('Prediction : ')
disp(predictions);
disp(['Correctly classified measurements: ' num2str(correct_count)]);

