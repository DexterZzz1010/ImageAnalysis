function predictedLabels = KNN_classify(testFeatures, knnClassifier)
    % Use the trained k-NN classifier to classify the test data
    % testFeatures = zscore(testFeatures');
        % 计算特征的最小值和最大值
    % Perform classification
    predictedLabels = predict(knnClassifier, testFeatures');
    predictedLabels
end
