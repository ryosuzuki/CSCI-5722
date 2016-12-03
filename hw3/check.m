%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ryo Suzuki
% rysu7393
% 105790212
% ryo.suzuki@colorado.edu
%
% CSCI-5722 Computer Vision
% Ioana Fleming
% Homework Assignment 3
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [binaryMap] = ssd(disparityLR, disparityRL, threshold)
  [row, col] = size(disparityLR);
  binaryMap = zeros(row, col);

  for (i = 1:row)
    for (j = 1:col)
      x_r = j - disparityLR(i, j);
      if (x_r < 1 || col < x_r)
        binaryMap(i, j) = 0;
      else
        d = abs(disparityLR(i, j) - disparityRL(i, x_r));
        if (d > threshold)
          binaryMap(i, j) = 255; % outlier
        else
          binaryMap(i, j) = 0; % inlier
        end
      end
    end
  end
end
