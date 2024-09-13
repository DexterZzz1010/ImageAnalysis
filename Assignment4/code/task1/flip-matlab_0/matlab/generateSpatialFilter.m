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

function s = generateSpatialFilter(PixelsPerDegree, channel)
    % Prepare Gaussian
    g =@(x, a1, b1, a2, b2) a1 * sqrt(pi / b1) * exp(-pi^2 * x / b1) + a2 * sqrt(pi / b2) * exp(- pi^2 * x / b2); % Square of x cancels with the sqrt in the distance calculation

    % Set parameters based on which channel the filter will be used for
    a1_A = 1; b1_A = 0.0047; a2_A = 0; b2_A = 1e-5;% avoid division by 0
    a1_rg = 1; b1_rg = 0.0053; a2_rg = 0; b2_rg = 1e-5;% avoid division by 0
    a1_by = 34.1; b1_by = 0.04; a2_by = 13.5; b2_by = 0.025;
    if strcmp(channel, 'A') % Achromatic CSF
        a1 = a1_A;
        b1 = b1_A;
        a2 = a2_A;
        b2 = b2_A;
    elseif strcmp(channel, 'RG') % Red-Green CSF
        a1 = a1_rg;
        b1 = b1_rg;
        a2 = a2_rg;
        b2 = b2_rg;
    elseif strcmp(channel, 'BY') % Blue-Yellow CSF
        a1 = a1_by;
        b1 = b1_by;
        a2 = a2_by;
        b2 = b2_by;
    else
        disp('You can not filter this channel! Check generateSpatialFilter.m.');
    end

    % Determine evaluation domain
    maxScaleParameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by]);
    radius = ceil(3 * sqrt(maxScaleParameter / (2 * pi^2)) * PixelsPerDegree);
    [xx, yy] = ndgrid(-radius:radius, -radius:radius);
    deltaX = 1 / PixelsPerDegree;
    xx = xx * deltaX;
    yy = yy * deltaX;
    d = xx .* xx + yy .* yy;

    % Generate filter and normalize sum to 1
    s = g(d, a1, b1, a2, b2);
    s = s / sum(s(:));
end
