function net = BP_train(inputData, targetData)
    % Create a feedforward neural network

    inputData = zscore(inputData);

    P=inputData;%输入数据
    T=targetData;%输出数据
    net = newff(P,T,20);%建立BP神经网络 含20个隐藏神经元
    
    % Define training parameters
    net.trainFcn = 'trainlm'; % You can choose different training algorithms
    net.trainParam.epochs = 100; % Number of training epochs
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

end

