function intersection = lineIntersect(x1, x2, x3, x4)
    % 计算交点
    A = [x1(1) - x2(1), x3(1) - x4(1); x1(2) - x2(2), x3(2) - x4(2)];
    B = [x3(1) - x2(1); x3(2) - x2(2)];
    
    % 求解线性方程组
    intersection = linsolve(A, B);
end
