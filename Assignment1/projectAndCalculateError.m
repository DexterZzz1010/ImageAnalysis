function [up, r] = projectAndCalculateError(u, basis)
    % Flatten the image into a column vector
    reshape_u = u(:) ; 
    % Create a matrix containing the basis vectors as columns
    reshape_basis = reshape(basis, [], 4);
    x = reshape_basis \ reshape_u;
    up = reshape_basis * x;    
    % Calculate the error norm
    r = norm(reshape_u - up,"fro");
%     r = sum(abs(reshape_u - up).^2);
    up = reshape(up, 19, 19);
end
