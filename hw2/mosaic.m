%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ryo Suzuki
% rysu7393
% 105790212
% ryo.suzuki@colorado.edu
%
% CSCI-5722 Computer Vision
% Ioana Fleming
% Homework Assignment 2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function mosaic_img = mosaic(img_1, img_2, homography)
  mosaic_img = img_1;

  % Add offsets for a larger image
  mosaic_img = padarray(mosaic_img, [0 size(img_2, 2)], 0, 'post');
  mosaic_img = padarray(mosaic_img, [size(img_2, 1) 0], 0, 'both');

  % Conbine img_1 and img_2
  for i = 1:size(mosaic_img, 2)
    for j = 1:size(mosaic_img, 1)
      p2 = homography * [i; j-floor(size(img_2, 1)); 1];
      p2 = p2 ./ p2(3);

      x2 = floor(p2(1));
      y2 = floor(p2(2));

      if x2 > 0 && x2 <= size(img_2, 2) && y2 > 0 && y2 <= size(img_2, 1)
        mosaic_img(j, i, :) = img_2(y2, x2, :);
      end
    end
  end

  % Crop images
  [row,col] = find(mosaic_img);
  c = max(col(:));
  d = max(row(:));

  st=imcrop(mosaic_img, [1 1 c d]);

  [row,col] = find(mosaic_img ~= 0);
  a = min(col(:));
  b = min(row(:));
  st=imcrop(st, [a b size(st,1) size(st,2)]);

  mosaic_img = st;


