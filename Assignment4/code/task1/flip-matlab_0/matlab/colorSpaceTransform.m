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

function transformedColor = colorSpaceTransform(inputColor, fromSpace2toSpace, firstInChain, lastInChain)
    % Check if inputColor is first part of transform chain. If so,
    % transform layout
    dim = size(inputColor);
    if firstInChain
        % Transform HxWx3 image to 3xHW for easier processing
        inputColor = reshape(permute(inputColor, [3, 1, 2]), dim(3), dim(1) * dim(2));
    end

    if strcmp(fromSpace2toSpace, 'srgb2linrgb')
        limit = 0.04045;
        allAboveLimit = inputColor > limit;
        transformedColor = zeros(size(inputColor));
        transformedColor(allAboveLimit) = ((inputColor(allAboveLimit) + 0.055) / 1.055) .^ 2.4;
        transformedColor(~allAboveLimit) = inputColor(~allAboveLimit) / 12.92;

    elseif strcmp(fromSpace2toSpace, 'linrgb2srgb')
        limit = 0.0031308;
        allAboveLimit = inputColor > limit;
        transformedColor = zeros(size(inputColor));
        transformedColor(allAboveLimit) = 1.055 * inputColor(allAboveLimit) .^ (1.0 / 2.4) - 0.055;
        transformedColor(~allAboveLimit) = 12.92 * inputColor(~allAboveLimit);

    elseif strcmp(fromSpace2toSpace, 'linrgb2xyz') || strcmp(fromSpace2toSpace, 'xyz2linrgb')
        % Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        % Assumes D65 standard illuminant
        a11 = 10135552 / 24577794;
        a12 = 8788810 / 24577794;
        a13 = 4435075 / 24577794;
        a21 = 2613072 / 12288897;
        a22 = 8788810 / 12288897;
        a23 = 887015 / 12288897;
        a31 = 1425312 / 73733382;
        a32 = 8788810 / 73733382;
        a33 = 70074185 / 73733382;
        A = [a11 a12 a13;
             a21 a22 a23;
             a31 a32 a33];
        if strcmp(fromSpace2toSpace, 'linrgb2xyz')
            transformedColor = A * inputColor;
        else
            transformedColor = A \ inputColor;
        end

    elseif strcmp(fromSpace2toSpace, 'xyz2ycxcz')
        reference_illuminant = colorSpaceTransform(cat(3, 1, 1, 1), 'linrgb2xyz', 1, 1);
        reference_illuminant = reshape(permute(reference_illuminant, [3, 1, 2]), 3, 1);
        reference_illuminant_matrix = repmat(reference_illuminant, [1, numel(inputColor) / 3]);
        inputColor = inputColor ./ reference_illuminant_matrix;
        Y = 116 * inputColor(2, :) - 16;
        Cx = 500 * (inputColor(1,:) - inputColor(2, :));
        Cz = 200 * (inputColor(2, :) - inputColor(3, :));
        transformedColor = [Y;Cx;Cz];

    elseif strcmp(fromSpace2toSpace, 'ycxcz2xyz')
        Yy = (inputColor(1, :) + 16) / 116;
        Cx = inputColor(2,:) / 500;
        Cz = inputColor(3,:) / 200;

        X = Yy + Cx;
        Y = Yy;
        Z = Yy - Cz;
        transformedColor = [X;Y;Z];

        reference_illuminant = colorSpaceTransform(cat(3, 1, 1, 1), 'linrgb2xyz', 1, 1);
        reference_illuminant = reshape(permute(reference_illuminant, [3, 1, 2]), 3, 1);
        reference_illuminant_matrix = repmat(reference_illuminant, [1, numel(transformedColor) / 3]);
        transformedColor = transformedColor .* reference_illuminant_matrix;
    elseif strcmp(fromSpace2toSpace, 'xyz2lab')
        reference_illuminant = colorSpaceTransform(cat(3, 1, 1, 1), 'linrgb2xyz', 1, 1);
        reference_illuminant = reshape(permute(reference_illuminant, [3, 1, 2]), 3, 1);
        reference_illuminant_matrix = repmat(reference_illuminant, [1, numel(inputColor) / 3]);
        inputColor = inputColor ./ reference_illuminant_matrix;
        delta = 6 / 29;
        limit = 0.008856;
        allAboveLimit = inputColor > limit;
        inputColor(allAboveLimit) = inputColor(allAboveLimit).^(1/3);
        inputColor(~allAboveLimit) = inputColor(~allAboveLimit) / (3 * delta * delta) + 4 / 29;
        L = 116 * inputColor(2, :) - 16;
        a = 500 * (inputColor(1,:) - inputColor(2, :));
        b = 200 * (inputColor(2, :) - inputColor(3, :));
        transformedColor = [L;a;b];

    elseif strcmp(fromSpace2toSpace, 'lab2xyz')
        L = (inputColor(1, :) + 16) / 116;
        a = inputColor(2,:) / 500;
        b = inputColor(3,:) / 200;

        X = L + a;
        Y = L;
        Z = L - b;
        transformedColor = [X;Y;Z];

        delta = 6/29;
        allAboveDelta = transformedColor > delta;
        transformedColor(allAboveDelta) = transformedColor(allAboveDelta).^3;
        transformedColor(~allAboveDelta) = 3 * delta^2 * (transformedColor(~allAboveDelta) - 4 / 29);

        reference_illuminant = colorSpaceTransform(cat(3, 1, 1, 1), 'linrgb2xyz', 1, 1);
        reference_illuminant = reshape(permute(reference_illuminant, [3, 1, 2]), 3, 1);
        reference_illuminant_matrix = repmat(reference_illuminant, [1, numel(transformedColor) / 3]);
        transformedColor = transformedColor .* reference_illuminant_matrix;

    elseif strcmp(fromSpace2toSpace, 'srgb2xyz')
        transformedColor = colorSpaceTransform(inputColor, 'srgb2linrgb', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2xyz', 0, 0);
        lastInChain = 1;
    elseif strcmp(fromSpace2toSpace, 'srgb2ycxcz')
        transformedColor = colorSpaceTransform(inputColor, 'srgb2linrgb', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2xyz', 0, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2ycxcz', 0, 0);
        lastInChain = 1;
    elseif strcmp(fromSpace2toSpace, 'linrgb2ycxcz')
        transformedColor = colorSpaceTransform(inputColor, 'linrgb2xyz', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2ycxcz', 0, 0);
        lastInChain = 1;
    elseif strcmp(fromSpace2toSpace, 'srgb2lab')
        transformedColor = colorSpaceTransform(inputColor, 'srgb2linrgb', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2xyz', 0, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2lab', 0, 0);
        lastInChain = 1;
    elseif strcmp(fromSpace2toSpace, 'linrgb2lab')
        transformedColor = colorSpaceTransform(inputColor, 'linrgb2xyz', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2lab', 0, 0);
        lastInChain = 1;
    elseif strcmp(fromSpace2toSpace, 'ycxcz2linrgb')
        transformedColor = colorSpaceTransform(inputColor, 'ycxcz2xyz', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2linrgb', 0, 0);
        lastInChain = 1;
    elseif strcmp(fromSpace2toSpace, 'lab2srgb')
        transformedColor = colorSpaceTransform(inputColor, 'lab2xyz', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2linrgb', 0, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2srgb', 0, 0);
        lastInChain = 1;
    elseif strcmp(fromSpace2toSpace, 'ycxcz2lab')
        transformedColor = colorSpaceTransform(inputColor, 'ycxcz2xyz', 1, 0);
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2lab', 0, 0);
        lastInChain = 1;
    else
        disp('The color transform is not defined!'); disp(fromSpace2toSpace)
        transformedColor = inputColor;
    end

    % Transform back to HxWx3 layout if transform is last in chain (HxWx1 in case of grayscale)
    if lastInChain
        transformedColor = permute(reshape(transformedColor, size(transformedColor, 1), dim(1), dim(2)), [2, 3, 1]);
    end
end

