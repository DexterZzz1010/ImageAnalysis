function roundedPredictions = CNN_classify(x, classification_data)
    % x是测试集，大小为361*n，每列是一张19*19的图片

    net = classification_data;
    roundedPredictions = 0 ;
    predictions = net(x);  
    roundedPredictions = round(predictions);
    % Set any negative predictions to zero
    roundedPredictions(roundedPredictions < 1) = 1;
    roundedPredictions(roundedPredictions > 10) = 10;
    roundedPredictions
%     [~, predictions] = max(predictions, [], 2);
%     predictions
end

