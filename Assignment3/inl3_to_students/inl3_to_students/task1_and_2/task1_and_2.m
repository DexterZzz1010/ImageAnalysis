% SCRIPT FOR BOTH TASK 1 AND 2.
%
% Task 1: Create your own classifier and try it out by classifying images
% as faces or non-faces
%
% Task 2: Try some built-in machine learning methods for classifying these
% same images of faces / non-faces

% Clear up
clc;
close all;
clearvars;
warning off

% Begin by loading the data
load FaceNonFace
nbr_examples = length(Y);

% This outer loop will run 100 times, so that you get a mean error for your
% classifier (the error will become different each time due to the
% randomness of cvpartition, which you may verify if you wish).
nbr_trials = 1;
err_rates_test = zeros(nbr_trials, 4);
err_rates_train = zeros(nbr_trials, 4);
for i = 1 : nbr_trials
    
    % First split data into training / testing (80% train, 20% test)
    part = cvpartition(nbr_examples, 'HoldOut', 0.20);
    
    % Extract training and test data given the partition above
    X_train = X(:, part.training);
    X_test = X(:, part.test); % 提取测试数据

    Y_train = Y(:,part.training); % 提取训练标签
    Y_test = Y(:,part.test); % 提取测试标签
    nbr_train_examples = length(Y_train);
    nbr_test_examples = length(Y_test);
    
    % Now we can train our model!
    % YOU SHOULD IMPLEMENT THE FUNCTION class_train!
    classification_data = class_train(X_train, Y_train);
        
    % Next, let's use our trained model to classify the examples in the 
    % test data
    predictions_test = zeros(1, nbr_test_examples);
    for j = 1 : nbr_test_examples
        % YOU SHOULD IMPLEMENT THE FUNCTION classify!
        predictions_test(j) = classify(X_test(:, j), classification_data);
    end
   
    % We do the same thing again but this time for the training data itself!
    predictions_train = zeros(1, nbr_train_examples);
    for j = 1 : nbr_train_examples
        % YOU SHOULD IMPLEMENT THE FUNCTION classify!
        predictions_train(j) = classify(X_train(:, j), classification_data);
    end
    
    % We can now proceed to computing the respective error rates.
    pred_test_diff = predictions_test - Y_test;
    pred_train_diff = predictions_train - Y_train;
    err_rate_test = nnz(pred_test_diff) / nbr_test_examples;
    err_rate_train = nnz(pred_train_diff) / nbr_train_examples;
    
    % Store them in the containers
    err_rates_test(i, 1) = err_rate_test;
    err_rates_train(i, 1) = err_rate_train;
    
    %--------------------- Below: Task 2 ------------------------------%
    % 
    % Uncomment the below code when working on Task 2
    % NOTE: Running the full script can take several minutes. However, by
    % setting nbr_trials = 1 (see almost at the top of this script) when
    % trying things it will go faster.
    % 
    % DO NOT FORGET TO SET nbr_trials = 100 AGAIN WHEN REPORTING YOUR FINAL
    % RESULTS
    
    % Train built-in functions (don't forget: transpose as necessary)
    tree_model = fitctree(X_train', Y_train);  % 举例：使用 fitctree 训练决策树模型
    svm_model = fitcsvm(X_train', Y_train);  % 举例：使用 fitcsvm 训练SVM模型
    nn_model =fitcknn(X_train', Y_train);
    % nn_model = trainNeuralNetwork(X_train, Y_train);  % 举例：使用 trainNeuralNetwork 训练神经网络模型

    
    % Next, let's use our trained model to classify the examples in the 
    % test data. You should look up the function "predict" in Matlab!
    % (don't forget: transpose as necessary, both for X and Y)
    predictions_test_tree = predict(tree_model, X_test');  % 使用决策树模型进行测试数据的分类
    predictions_test_svm = predict(svm_model, X_test');  % 使用SVM模型进行测试数据的分类
    predictions_test_nn = predict(nn_model, X_test');  
   
    % We can now proceed to computing the respective error rates.
    pred_test_diff_tree = predictions_test_tree' - Y_test;
    pred_test_diff_svm = predictions_test_svm' - Y_test;
    pred_test_diff_nn = predictions_test_nn' - Y_test;
    err_rate_test_tree = nnz(pred_test_diff_tree) / length(Y_test);
    err_rate_test_svm = nnz(pred_test_diff_svm) / length(Y_test);
    err_rate_test_nn = nnz(pred_test_diff_nn) / length(Y_test);
    
    % Store them in the containers
    err_rates_test(i, 2) = err_rate_test_tree;
    err_rates_test(i, 3) = err_rate_test_svm;
    err_rates_test(i, 4) = err_rate_test_nn;
    
    % Let's do the same for the training data
    predictions_train_tree = predict(tree_model, X_train');  % 使用决策树模型进行训练数据的分类
    predictions_train_svm = predict(svm_model, X_train');  % 使用SVM模型进行训练数据的分类
    predictions_train_nn = predict(nn_model, X_train');  

    pred_train_diff_tree = predictions_train_tree' - Y_train;
    pred_train_diff_svm = predictions_train_svm' - Y_train;
    pred_train_diff_nn = predictions_train_nn' - Y_train;
    err_rate_train_tree = nnz(pred_train_diff_tree) / numel(Y_train);  % 计算决策树模型的错误率
    err_rate_train_svm = nnz(pred_train_diff_svm) / numel(Y_train);  % 计算SVM模型的错误率
    err_rate_train_nn = nnz(pred_train_diff_nn) / numel(Y_train); 

    % Store them in the containers for training data
    err_rates_train(i, 2) = err_rate_train_tree;
    err_rates_train(i, 3) = err_rate_train_svm;
    err_rates_train(i, 4) = err_rate_train_nn;
end

% Finally, after all the trials are done, report mean error rates
%
% NOTE: From here you get two 4-dimensional arrays, where the first entry
% is the mean error rate for your own method, and the last three
% entries are mean error rates for the built-in methods
%
% DO NOT FORGET: COMMENT ON ALL RESULTS IN THE REPORT
%
mean_err_rate_test = mean(err_rates_test, 1)
mean_err_rate_train = mean(err_rates_train, 1)

% ------------------ FOR TASK 1 BELOW! ---------------------------------

% For the report, in addition to reporting above error rates (and
% commenting on them!), you should also manually select two images from X,
% one image of a face, and one image of a non-face. Begin by reshaping each
% such column from 361x1 to 19x19, so that you can show them using matlab's
% function "imagesc". (See "reshape" in Matlab for how to reshape a vector to a
% matrix). Provide both these images in your report INCLUDING WHAT YOUR
% MODEL PREDICTS THE RESPECTIVE IMAGES TO BE. Hopefully your model will say
% that the face is a face, and that the non-face is a non-face, but it's
% not certain of course. Make sure that the two images are extracted from
% X_test, and not X_train. Write code for this below! 

 % Loop through all test examples
count_pos = 0;
count_neg = 0;
    for j = 10 : nbr_test_examples
        predictions_test(j) = classify(X_test(:, j), classification_data);
        % If prediction is 1 and we haven't displayed two such images yet
        if predictions_test(j) == 1 && count_pos < 2
            count_pos = count_pos + 1;
            % Reshape the image and display it
            img = reshape(X_test(:, j), [19, 19]);
            figure;
            imagesc(img);
            title(['Face ', num2str(count_pos)]);
        % If prediction is -1 and we haven't displayed two such images yet
        elseif predictions_test(j) == -1 && count_neg < 2
            count_neg = count_neg + 1;
            % Reshape the image and display it
            img = reshape(X_test(:, j), [19, 19]);
            figure;
            imagesc(img);
            title(['Non-Face ', num2str(count_neg)]);
        elseif count_pos >= 2 && count_neg >= 2
            break;
        end
        
    end