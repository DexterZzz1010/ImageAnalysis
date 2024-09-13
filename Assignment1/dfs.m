% Function to perform DFS for connected components labeling
function dfs(pixel,i, j,current_label,binary_image,labeled_img,labeled_visited_coordinates,neighbors,connected_components)
    
    global labeled_visited_coordinates
    global labeled_img
    global connected_components
    
    if labeled_visited_coordinates(i, j) == 0
        labeled_visited_coordinates(i, j) = 1;
        if binary_image(i,j)==pixel
            labeled_img(i, j) = current_label;
            connected_components{current_label} = [connected_components{current_label}; [i, j]];
                
            % Recursively call DFS on neighboring pixels
            for k = 1:size(neighbors, 1)
                ni = i + neighbors(k, 1);
                nj = j + neighbors(k, 2);
                if ni < 1 || ni > size(binary_image, 1) || nj < 1 || nj > size(binary_image, 2) 
                    continue;
                else
                    dfs(pixel,ni, nj,current_label,binary_image,labeled_img,labeled_visited_coordinates,neighbors,connected_components);
                end
            end    
        end    
    end   
    
end

