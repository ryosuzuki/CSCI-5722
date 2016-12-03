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

function [disparityMap] = ssd(frameLeftGray, frameRightGray, windowSize, disparityRange)
  frameLeftGray = im2double(frameLeftGray);
  frameRightGray = im2double(frameRightGray);

  disparityMin = disparityRange(1);
  disparityMax = disparityRange(2);

  [row, col] = size(frameLeftGray);
  % [row, col] = size(frameRightGray);

  disparityMap = zeros(row, col);
  weight = fspecial('gaussian', windowSize, 1);
  pad = (windowSize-1)/2;

  for (i = 1+pad:row-pad)
    for (j = 1+pad:col-pad-disparityMax)
      prevSSD = Inf;
      bestMatch = disparityMin;
      for (d = disparityMin:disparityMax)
        ssd = 0.0;
        for (a = -pad:pad)
          for (b = -pad:pad)
            if (j+b+d <= col)
              ssd = ssd + ( weight(a+pad+1,b+pad+1) * (frameRightGray(i+a,j+b)-frameLeftGray(i+a,j+b+d)) )^2;
            end
          end
        end
        if (prevSSD > ssd)
          prevSSD = ssd;
          bestMatch = d;
        end
      end
      disparityMap(i,j) = bestMatch;
    end
  end
end
