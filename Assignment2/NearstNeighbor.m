clc
clear
close all

% Define class measurements and labels
class1_measurements = [0.4003, 0.3988, 0.3998, 0.3997];
class2_measurements = [0.2554, 0.3139, 0.2627, 0.3802];
class3_measurements = [0.5632, 0.7687, 0.0524, 0.7586];
class_labels = [1, 2, 3]; % Corresponding class labels

% Define test measurements
test_measurements = [
    [0.4010, 0.3995, 0.3991]; % Test data for Class 1
    [0.3287, 0.3160, 0.2924]; % Test data for Class 2
    [0.4243, 0.5005, 0.6769]  % Test data for Class 3
];

% Initialize counter for correct classifications
correct_classifications = 0;

% Loop through each test measurement
for i = 1:size(test_measurements, 1)
    test_measurement = test_measurements(i, :);   
    for p = 1:length(test_measurement)
        % Initialize variables for nearest neighbor search
        nearest_class = 0;
        min_distance = Inf;
        % Loop through training measurements in each class
        for j = 1:numel(class_labels)
            class_label = class_labels(j);
            
            % Get the training measurements for the current class
            train_measurements = [];
            if class_label == 1
                train_measurements = class1_measurements;
            elseif class_label == 2
                train_measurements = class2_measurements;
            elseif class_label == 3
                train_measurements = class3_measurements;
            end
            
            for k = 1:length(train_measurements)
                % Calculate distance between the test measurement and each training measurement
                distance = abs(train_measurements(k) - test_measurement(p));
                
                % Find the minimum distance and corresponding class label
                if distance < min_distance
                    min_distance = distance;
                    nearest_class = class_label;
                end
            end
        end
        
        % Check if the nearest neighbor classification is correct
        if nearest_class == i
            correct_classifications = correct_classifications + 1;
        end
    end  
end

% Display the number of correctly classified test measurements
disp(['Correctly classified measurements: ' num2str(correct_classifications)]);
