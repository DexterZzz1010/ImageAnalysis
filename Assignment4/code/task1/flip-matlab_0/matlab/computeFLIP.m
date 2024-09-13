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

function [error,deltaE] = computeFLIP(reference, test, PixelsPerDegree)
    if nargin<3
        PixelsPerDegree = 67;
    end

    assert(isequal(size(reference), size(test)), 'Reference and test not of equal size.')

    % Set color and feature exponents
    qc = 0.7;
    qf = 0.5;
    
    % Transform reference and test to opponent color space
    reference = colorSpaceTransform(reference, 'srgb2ycxcz', 0, 0);
    test = colorSpaceTransform(test, 'srgb2ycxcz', 0, 0);
    
    % --- Color pipeline ---
    % Spatial filtering
    s_a = generateSpatialFilter(PixelsPerDegree, 'A');
    s_rg = generateSpatialFilter(PixelsPerDegree, 'RG');
    s_by = generateSpatialFilter(PixelsPerDegree, 'BY');
    filteredReference = spatialFilter(reference, s_a, s_rg, s_by);
    filteredTest = spatialFilter(test, s_a, s_rg, s_by);

    % Perceptually Uniform Color Space
    preprocessedReference = huntAdjustment(colorSpaceTransform(filteredReference, 'linrgb2lab', 0, 0));
    preprocessedTest = huntAdjustment(colorSpaceTransform(filteredTest, 'linrgb2lab', 0, 0));
    
    % Color metric
    deltaEhyab = HyAB(preprocessedReference, preprocessedTest);
    huntAdjustedGreen = huntAdjustment(colorSpaceTransform(cat(3, 0, 1, 0), 'linrgb2lab', 0, 0));
    huntAdjustedBlue = huntAdjustment(colorSpaceTransform(cat(3, 0, 0, 1), 'linrgb2lab', 0, 0));
    cmax = (HyAB(huntAdjustedGreen, huntAdjustedBlue)) ^ qc;
    deltaEc = redistributeErrors(deltaEhyab .^ qc, cmax);
    
    % --- Feature pipeline ---
    % Extract and normalize achromatic component
    referenceY = (reference(:, :, 1) + 16) / 116;
    testY = (test(:, :, 1) + 16) / 116;

    % Edge and point detection
    edgesReference = featureDetection(referenceY, PixelsPerDegree, 'edge');
    pointsReference = featureDetection(referenceY, PixelsPerDegree, 'point');
    edgesTest = featureDetection(testY, PixelsPerDegree, 'edge');
    pointsTest = featureDetection(testY, PixelsPerDegree, 'point');

    % Feature metric
    deltaEf = (1 / sqrt(2) * ...
               max(abs(vecnorm(edgesReference, 2, 3) - vecnorm(edgesTest, 2, 3)), ...
                   abs(vecnorm(pointsReference, 2, 3) - vecnorm(pointsTest, 2, 3)))) .^ qf;

    % --- Final error ---
    deltaE = deltaEc .^ (1 - deltaEf);
    error = mean(abs(deltaE(:)));
    
end
