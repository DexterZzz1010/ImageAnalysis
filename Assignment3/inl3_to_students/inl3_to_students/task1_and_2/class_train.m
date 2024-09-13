function classification_data = class_train(X, Y)
    % Training
    % X是训练集，大小为361*200，每列是一张19*19的图片
    % Y是标签，大小为200*1，取值为1或-1

     % 获取特征数量和样本数量
    [n, m] = size(X);

    % 初始化变量
    alpha = zeros(1, m);
    b = 0;
    lr = 0.01;  % 学习率
    num_iterations = 1000;  % 迭代次数

    % 训练 SVM 模型
    for iteration = 1:num_iterations
        for i = 1:m
            % 计算预测值
            y_pred = b;
            for j = 1:m
                y_pred = y_pred + alpha(j) * Y(j) * dot(X(:, i), X(:, j));
            end

            % 更新 alpha
            if Y(i) * y_pred < 1
                alpha(i) = alpha(i) + lr;

            end
        end
    end

    % 计算权重向量
    w = zeros(n, 1);
    for i = 1:m
        w = w + alpha(i) * Y(i) * X(:, i);
    end

    % 计算偏置项
    b = mean(Y - w' * X);

    % 返回 SVM 模型
    svm_model.w = w;
    svm_model.b = b;

    % 返回classification_data，包含SVN模型信息
    classification_data.svm_model= svm_model;
end
