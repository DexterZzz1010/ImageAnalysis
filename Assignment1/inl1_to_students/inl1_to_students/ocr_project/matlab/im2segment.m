function S = im2segment(img)
img = uint8(img);
% figure;
% imshow(img);

% Specify the standard deviation of the Gaussian filter (to control the degree of blurring)
sigma = 0.5;

% Perform Gaussian filtering
img = imgaussfilt(img, sigma);

% img = imbilatfilt(img);
% img = medfilt2(img, [3, 3])

threshold = 0.158 % Set the threshold of image binarize
binary_image = imbinarize(img, threshold); % binarize
% imshow(binary_image);

%% 8-connected components
labeledImage = bwlabel(binary_image, 8);
minPixels = 1 ; % set min pixel num


%% Extracting information about connected components
stats = regionprops(labeledImage, 'BoundingBox', 'PixelIdxList');

% Initialize an array of cells for storing segmented images
numStats = numel(stats);
S = cell(1, numStats);

% Create an array of flags to keep track of merged cells
merged = false(1, numStats);
small = false(1, numStats);

%% Storing coordinate indexes of different labels in cells
for i = 1:numStats
    % Get the coordinate index of the current connected component
    pixelIdxList = stats(i).PixelIdxList;

    % Create a segmented image of the same size as the original image
    segmented_image = zeros(size(labeledImage));
    segmented_image(pixelIdxList) = 1;
    numPixels = sum(segmented_image(:) == 1);
    if numPixels < minPixels
        small(i)=true;
    end
    % Storing Segmented Images into Cells
    S{i} = segmented_image;
end


%% 
Dis_threshold = 20; % distance threshold

% Calculate the center coordinates of each cell and combine them
for i = 1:numStats
    if ~merged(i) && ~small(i)
        for j = i+1:numStats
            if ~merged(j) && ~small(j)
                % Calculate the center coordinates
                centro_i = regionprops(S{i}, 'Centroid');
                centro_j = regionprops(S{j}, 'Centroid');
                
                % Extracting the center coordinates
                centro_i = centro_i.Centroid;
                centro_j = centro_j.Centroid;
                
                % culculate the Euclidean distance
                distance = norm(centro_i - centro_j);
                
                if distance < Dis_threshold
                    % Merge two cells and add elements to the first cell
                    S{i} = S{i} | S{j};
                    merged(j) = true; 
                end
            end
        end
    end
end

S = S(~merged);
end
