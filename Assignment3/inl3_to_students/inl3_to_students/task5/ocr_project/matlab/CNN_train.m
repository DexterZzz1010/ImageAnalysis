function net = CNN_train(inputData, targetData)
    % Create a feedforward neural network

    hiddenLayerSizes = [100,1];
    net = feedforwardnet(hiddenLayerSizes); % 10 neurons in the output layer for ten classes
    
    % Define training parameters
    net.trainFcn = 'trainlm'; % You can choose different training algorithms
    net.trainParam.epochs = 1000; % Number of training epochs
    net.trainParam.showWindow = true; % Disable training window display
    
    % Split the data into training and testing sets
    net.divideFcn = 'dividerand'; % Random division of data
    net.divideParam.trainRatio = 0.8; % 80% of data for training
    net.divideParam.valRatio = 0.2; % 20% of data for validation (optional)
    net.trainParam.max_fail = 20;
    net.input.processFcns = {'mapminmax'};
    net.output.processFcns = {'mapminmax'};
     net = configure(net,inputData,targetData);
    % Train the neural network
    % net = train(net, inputData, targetData);
    net_GPU = train(net, inputData, targetData,'useGPU','yes');
    net = gather(net_GPU);
%     predictions = sim(net,inputData);  
%     predictions
end

% function net = CNN_train(inputData, targetData)
%     % Create a feedforward neural network
%     net = feedforwardnet(10); % 10 neurons in the output layer for ten classes
%     
%     inputDataGPU = gpuArray(inputData);
%     targetDataGPU = gpuArray(targetData);
% 
%     % Define training parameters
%     net.trainFcn = 'trainlm'; % You can choose different training algorithms
%     net.trainParam.epochs = 100; % Number of training epochs
%     net.trainParam.showWindow = true; % Disable training window display
%     
%     % Split the data into training and testing sets
%     net.divideFcn = 'dividerand'; % Random division of data
%     net.divideParam.trainRatio = 0.8; % 80% of data for training
%     net.divideParam.valRatio = 0.2; % 20% of data for validation (optional)
%     
%     % Train the neural network
%     net = train(net, inputDataGPU, targetDataGPU);
%     net = gather(net);
% end
