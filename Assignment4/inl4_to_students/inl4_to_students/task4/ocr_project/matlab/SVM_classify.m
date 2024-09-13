function predictedLabels = SVM_classify(testFeatures,classifier )
    % Use the trained classifier to classify the test data
    
    % Optional: Normalize the feature data (make sure to use the same scaling as in training)
    %testFeatures = zscore(testFeatures');
    
    % Perform classification
    predictedLabels = predict(classifier, testFeatures');
end
