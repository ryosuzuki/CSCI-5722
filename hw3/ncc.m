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

function [disparityMap] = ncc(frameLeftGray, frameRightGray, windowSize, disparityRange)
  frameLeftGray = double(frameLeftGray);
  frameRightGray = double(frameRightGray);

  disparityMin = disparityRange(1);
  disparityMax = disparityRange(2);

  [row, col] = size(frameLeftGray);
  % [row, col] = size(frameRightGray);

  disparityMap = zeros(row, col);
  pad = (windowSize-1)/2;

  for (i = 1+pad:row-pad)
    for (j = 1+pad:col-pad-disparityMax)
      prevNCC = 0.0;
      bestMatch = disparityMin;
      for (d = disparityMin:disparityMax)
        ncc = 0.0;
        nccNumerator = 0.0;
        nccDenominator = 0.0;
        nccDenominatorRight = 0.0;
        nccDenominatorLeft = 0.0;
        for (a = -pad:pad)
          for (b = -pad:pad)
            nccNumerator = nccNumerator + (frameRightGray(i+a,j+b) * frameLeftGray(i+a,j+b+d));
            nccDenominatorRight = nccDenominatorRight + (frameRightGray(i+a,j+b) * frameRightGray(i+a,j+b));
            nccDenominatorLeft = nccDenominatorLeft + (frameLeftGray(i+a,j+b+d) * frameLeftGray(i+a,j+b+d));
          end
        end
        nccDenominator = sqrt(nccDenominatorRight*nccDenominatorLeft);
        ncc = nccNumerator/nccDenominator;
        if (prevNCC < ncc)
          prevNCC = ncc;
          bestMatch = d;
        end
      end
      disparityMap(i,j) = bestMatch;
    end
  end
end