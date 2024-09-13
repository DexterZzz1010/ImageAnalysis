%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%  * Redistributions of source code must retain the above copyright
%    notice, this list of conditions and the following disclaimer.
%  * Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the distribution.
%  * Neither the name of NVIDIA CORPORATION nor the names of its
%    contributors may be used to endorse or promote products derived
%    from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
% EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
% OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% FLIP: A Difference Evaluator for Alternating Images
% High Performance Graphics, 2020.
% by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller, Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild
%
% Pointer to our paper: https://research.nvidia.com/publication/2020-07_FLIP
% code by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller

function features = featureDetection(Iy, PixelsPerDegree, featureType)
    % Finds features of type featureType in image Iy based on current PPD
    
    % Set peak to trough value (2x standard deviations) of human edge
    % detection filter
    w = 0.082;
    
    % Compute filter radius
    sd = 0.5 * w * PixelsPerDegree;
    radius = ceil(3 * sd);

    % Compute 2D Gaussian
    [x, y] = meshgrid(-radius:radius, -radius:radius);
    g = exp(-(x.^2 + y.^2) / (2 * sd^2));
    
    if strcmp(featureType, 'edge') % Edge detector
        % Compute partial derivative in x-direction
        Gx = -x .* g;
    else % Point detector
        % Compute second partial derivative in x-direction
        Gx = (x .^ 2 / sd ^ 2 - 1) .* g;
    end
 
    % Normalize positive weights to sum to 1 and negative weights to sum to -1
    negativeWeightsSum = -sum(sum(Gx(Gx < 0)));
    positiveWeightsSum = sum(sum(Gx(Gx > 0)));
    Gx(Gx < 0) = Gx(Gx < 0) / negativeWeightsSum;
    Gx(Gx > 0) = Gx(Gx > 0) / positiveWeightsSum;
    
    % Symmetry yields the y-direction filter
    Gy = Gx';
    
    % Detect features
    featuresX = imfilter(Iy, Gx, 'conv', 'replicate');
    featuresY = imfilter(Iy, Gy, 'conv', 'replicate');
    features = cat(3, featuresX, featuresY);
end