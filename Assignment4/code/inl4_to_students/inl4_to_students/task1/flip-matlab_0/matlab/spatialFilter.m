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

function ITildeLinearRGB = spatialFilter(I, s_a, s_rg, s_by)
    % Filters image I using Contrast Sensitivity Functions.
    % Returns linear RGB

    % Apply Gaussian filters
    ITildeOpponent(:, :, 1) = imfilter(I(:, :, 1), s_a, 'conv', 'replicate');
    ITildeOpponent(:, :, 2) = imfilter(I(:, :, 2), s_rg, 'conv', 'replicate');
    ITildeOpponent(:, :, 3) = imfilter(I(:, :, 3), s_by, 'conv', 'replicate');

    % Transform to linear RGB for clamp
    ITildeLinearRGB = colorSpaceTransform(ITildeOpponent, 'ycxcz2linrgb', 0, 0);
    
    % Clamp to RGB box
    ITildeLinearRGB = max(min(ITildeLinearRGB, 1), 0);
end
