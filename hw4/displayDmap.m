%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ryo Suzuki
% rysu7393
% 105790212
% ryo.suzuki@colorado.edu
%
% CSCI-5722 Computer Vision
% Ioana Fleming
% Homework Assignment 4
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [coloredDisparityMap] = displayDmap(disparityMap)

  height = size(disparityMap,1);
  width = size(disparityMap,2);

  coloredDisparityMap = zeros(height,width,3);
  maxD = nanmax(nanmax(disparityMap));
  minD = nanmin(nanmin(disparityMap));

  for y = 1:height
    for x = 1:width
      if isnan(disparityMap(y,x))
        coloredDisparityMap(y,x,:) = [1,0,0];
      else
        coloredDisparityMap(y,x,:) = (disparityMap(y,x) - minD)./(maxD-minD);
      end
    end
  end

end