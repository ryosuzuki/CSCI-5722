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


function [disparityMap, D, b] = stereoD(image1, image2, maxDisp, occ)

  image1 = im2double(image1);
  image2 = im2double(image2);

  height = size(image1, 1);
  width = size(image1, 2);

  disparityMap = zeros(height,width);

  D = zeros(width,width,height);
  b = zeros(width,width,height);

  D(1,1,:) = 0;
  b(1,1,:) = 3;

  for y = 1:height
    disp(y)
    % disp(y/height)

    for x = 2:width
      D(1,x,y) = x*occ;
      if D(1,x,y) > D(1,x-1,y)
        b(1,x,y) = b(1,x,y);
      else
        b(1,x,y) = b(1,x-1,y);
      end

      D(x,1,y) = x*occ;
      if b(x,1,y) > b(x-1,1,y)
        b(x,1,y) = b(x-1,1,y);
      else
        b(x,1,y) = b(x-1,1,y);
      end
    end

    for i = 2:width
      for j = 2:width

        p1 = image1(y,i);
        p2 = image2(y,j);
        d_ij = (p1 - p2)^2;

        D_1 = D(i-1,j-1,y) + d_ij;
        D_2 = D(i-1,j,y) + occ;
        D_3 = D(i,j-1,y) + occ;

        if D_3 < D_1 && D_3 < D_2
          D(i,j,y) = D_3;
          b(i,j,y) = 1; % up
        elseif D_2 < D_1
          D(i,j,y) = D_2;
          b(i,j,y) = 2; % left
        else
          D(i,j,y) = D_1;
          b(i,j,y) = 3; % up and left
        end
      end
    end

    i = width;
    j = width;
    while i > 0 && j > 0
      if b(i,j,y) == 1
        disparityMap(y,j) = NaN;
        j = j-1;
      elseif b(i,j,y) == 2
        i = i-1;
      else
        disparityMap(y,j) = abs(i-j);
        i = i-1;
        j = j-1;
      end
    end
  end
end