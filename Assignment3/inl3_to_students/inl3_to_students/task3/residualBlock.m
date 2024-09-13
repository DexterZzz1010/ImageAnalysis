function layers = residualBlock(numFilters, numBlocks, name)
    layers = [
        convolution2dLayer(3, numFilters, 'Padding', 1, 'Stride', 1, 'Name', [name 'conv1'])
        batchNormalizationLayer('Name', [name 'bn1'])
        reluLayer('Name', [name 'relu1'])
        convolution2dLayer(3, numFilters, 'Padding', 1, 'Stride', 1, 'Name', [name 'conv2'])
        batchNormalizationLayer('Name', [name 'bn2'])
    ];
    
    shortcutLayers = [
        convolution2dLayer(1, numFilters, 'Padding', 0, 'Stride', 1, 'Name', [name 'shortcut_conv'])
        batchNormalizationLayer('Name', [name 'shortcut_bn'])
    ];
    
    layers = [
        layers
        additionLayer(2, 'Name', [name 'add'])
        reluLayer('Name', [name 'relu2'])
    ];
    
    layers = [
        shortcutLayers
        layers
    ];
    
    for i = 2:numBlocks
        layers = [
            convolution2dLayer(3, numFilters, 'Padding', 1, 'Stride', 1, 'Name', [name 'conv' num2str(i*2-1)])
            batchNormalizationLayer('Name', [name 'bn' num2str(i*2-1)])
            reluLayer('Name', [name 'relu' num2str(i*2-1)])
            convolution2dLayer(3, numFilters, 'Padding', 1, 'Stride', 1, 'Name', [name 'conv' num2str(i*2)])
            batchNormalizationLayer('Name', [name 'bn' num2str(i*2)])
        ];
        
        layers = [
            layers
            additionLayer(2, 'Name', [name 'add' num2str(i)])
            reluLayer('Name', [name 'relu' num2str(i*2)])
        ];
    end
end
