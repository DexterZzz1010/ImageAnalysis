clc;
clearvars;
close all;

% This script is an overview of what needs to be done prior to running any
% of the other scripts, e.g. inl3_test_and_benchmark.m. You
% of course need to change some things below (e.g. replace
% 'class_train' to whatever your class train function is called!)
% Also, make sure that all functions being called reside in this same
% folder, so that MATLAB finds them appropriately.

% First step in this script is to load ocrsegments for training. NOTE: Each
% single example in ocrsegments is used for training. We will have some
% other data to test on later. Thus we will not use cvpartition in this
% assignment. Two things will be loaded: S, containing 100 letter segments,
% and y, containing the 100 associated labels (valued 1 through 26).
load ocrsegments

% The next step is to take the each segment in S and transform into 
% feature vectors. We use segment2features for this, which is a function
% we implemented in hand-in 2.
S_feats = zeros(7, 100); % 7, since I use 7 features -- change appropriately!
for i = 1 : numel(S)
    S_feat = segment2features(S{i});
    S_feats(:, i) = S_feat;
end

% Now we are ready to train our classifier. We use our class_train.m that
% we wrote in Task 2 in this hand-in. I happened to name my as
% class_train_knn, instead (I implemented k-nearest neighbour), but of
% course rename to whatever you are using.
classification_data = class_train(S_feats, y);

% We can now save classification_data, since it will be loaded in some of
% the other functions used in this task (e.g. in inl3_test_and_benchmark.m).
save('classification_data.mat', 'classification_data')

% When this script has been successfully launched, we are ready to proceed.
% For instance, now you can try running inl3_test_and_benchmark.m