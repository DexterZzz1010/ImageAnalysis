function y = classify(x, classification_data)
    % x是测试集，大小为361*n，每列是一张19*19的图片
    % classification_data包含训练好的SVM模型

    % 提取SVM模型
    svm_model = classification_data.svm_model;

    % 使用SVM模型进行分类预测
    scores = svm_model.w' * x + svm_model.b;
    y = sign(scores);  % 根据预测分数进行二元分类，取值为1或-1
end