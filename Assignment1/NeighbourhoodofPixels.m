warning('off','all')
warning

% Define your binary image (replace this with your binary image data)
binary_image = [
    3 3 3 3 3 2 2 2 2 2 3 3 3 3 3;
    3 3 1 2 1 0 0 0 0 0 1 2 3 3 3;
    3 3 2 0 0 0 0 0 0 0 0 0 2 3 3;
    3 2 0 0 1 0 0 0 0 0 1 0 0 2 3;
    3 1 0 2 3 1 0 0 0 3 3 1 0 1 3;
    2 0 0 3 3 2 0 0 1 3 3 2 0 0 2;
    2 0 0 2 3 1 0 0 0 2 3 1 0 0 2;
    2 0 0 0 0 0 0 0 0 0 0 0 0 0 2;
    2 0 0 0 0 0 0 0 0 0 0 0 0 0 2;
    2 0 1 2 1 1 0 0 0 1 1 2 2 0 2;
    3 1 0 2 3 3 3 3 3 3 3 3 0 1 3;
    3 2 0 0 2 3 3 3 3 3 2 0 0 2 3;
    3 3 2 0 0 2 3 3 3 2 0 0 2 3 3;
    3 3 3 2 1 0 0 0 0 0 1 2 3 3 3;
    3 3 3 3 3 2 2 2 2 2 3 3 3 3 3;
];

pixel = 0;
% Define 8-connected neighbors
neighbors = [-1, -1; -1, 0; -1, 1; 0, 1; 1, 1; 1, 0; 1, -1; 0, -1];

% Initialize labeled image and label counter
global labeled_visited_coordinates
global labeled_img
global connected_components
labeled_visited_coordinates = zeros(size(binary_image));
labeled_img = zeros(size(binary_image));
current_label = 0;
connected_components = cell(0);
%% 
for i = 1:size(binary_image, 1)
    for j = 1:size(binary_image, 2)
        if binary_image(i, j) <=1
            binary_image(i, j) = 0;
        else
            binary_image(i, j) = 1;
        end 
    end
end
%%

% Iterate through the binary image to find and label connected components
for i = 1:size(binary_image, 1)
    for j = 1:size(binary_image, 2)
        if labeled_visited_coordinates(i, j)==0
            if binary_image(i, j) == pixel
                current_label = current_label + 1;
                connected_components{current_label} = [];
                dfs(pixel,i, j,current_label,binary_image,labeled_img,labeled_visited_coordinates,neighbors,connected_components);
            end
            labeled_visited_coordinates(i, j) = 1;
        end
    end
end

% Display the labeled image
imshow(label2rgb(labeled_img, 'jet', 'k'), 'InitialMagnification', 'fit');
title('Connected Components (8-connected)');
disp(labeled_img);

