% Given fundamental matrix F
F = [2, 2, 4; 3, 3, 6; -5, -10, -6];

% Point pair 1
a1 = [-4; 5; 1];
b1 = [3; 2; 1];
result1 = b1' * F * a1;

% Point pair 2
a2 = [3; -7; 1];
b2 = [6; -1; 1];
result2 = b2' * F * a2;

% Point pair 3
a3 = [-10; 5; 1];
b3 = [2; -2; 1];
result3 = b3' * F * a3;

% Display the results
disp('Result for point pair 1:');
disp(result1);

disp('Result for point pair 2:');
disp(result2);

disp('Result for point pair 3:');
disp(result3);