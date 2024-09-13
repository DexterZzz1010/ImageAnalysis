function knnClassifier = KNN_train(trainFeatures, trainLabels)
    % Train a k-NN classifier using the provided training data and k value
    k = 10;
    % trainFeatures = zscore(trainFeatures');
    % Create a k-NN classifier with the specified k value (you can adjust other parameters)
    knnClassifier = fitcknn(trainFeatures', trainLabels, 'NumNeighbors', k);
end

