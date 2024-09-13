clc
clear
close all
% Given images
u = [3 -7; -1 4];
v = 1/2 * [1 -1; -1 1];
w = 1/2 * [-1 1; -1 1];

% Calculate norms
norm_u = norm(u, 'fro');  % Frobenius norm for the image
norm_v = norm(v, 'fro');
norm_w = norm(w, 'fro');

% Calculate scalar products
u_dot_v = sum(sum(u .* v));
u_dot_w = sum(sum(u .* w));
v_dot_w = sum(sum(v .* w));

% Check if matrices u and v_dot_w are orthonormal
is_orthonormal = isequal(norm_v , 1) && isequal(norm_w , 1)&& isequal(dot(v(:), w(:)), 0);

% Calculate the orthogonal projection of u onto the subspace spanned by {v, w}
projection = (u_dot_v / (norm_v^2)) * v + (u_dot_w / (norm_w^2)) * w;

%%
approximation_error = sum(abs(u(:) - projection(:)).^2);
u_norm = (norm(u, 'fro'))^2;
abs_diff=abs(u(:) - projection(:));
diff_norm=(norm(abs_diff(:), 'fro'))^2;
diff = diff_norm/u_norm;

%%
% Display results
fprintf('Norm of u: %.2f\n', norm_u);
fprintf('Norm of v: %.2f\n', norm_v);
fprintf('Norm of w: %.2f\n', norm_w);
fprintf('Scalar Product u · v: %.2f\n', u_dot_v);
fprintf('Scalar Product u · w: %.2f\n', u_dot_w);
fprintf('Scalar Product v · w: %.2f\n', v_dot_w);
fprintf('Are matrices {v, w} orthonormal? %d\n', is_orthonormal);
disp('Orthogonal Projection of u onto {v, w}:');
disp(projection)
disp(['my diff: ',num2str(diff)]);
