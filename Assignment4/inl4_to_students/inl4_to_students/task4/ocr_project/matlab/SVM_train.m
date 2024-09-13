function classifier = SVM_train(trainFeatures, trainLabels)
    % Train an SVM classifier using the provided training data
    
    % Optional: Normalize the feature data (you can remove this if not needed)
    %trainFeatures = zscore(trainFeatures');
    
    % Create an SVM classifier (you can adjust the kernel and other parameters)
    classifier = fitcecoc(trainFeatures', trainLabels);
end