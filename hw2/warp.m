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

function warped_img = warp(img_2, homography)
  [X, Y, C] = size(img_2);

  % Set the bounding box
  bb = [1 Y 1 X];
  bb_xmin = bb(1);
  bb_xmax = bb(2);
  bb_ymin = bb(3);
  bb_ymax = bb(4);

  [U, V] = meshgrid(bb_xmin:bb_xmax, bb_ymin:bb_ymax);
  [nrows, ncols] = size(U);

  % Compute warped x- and y- coordinates
  u = U(:);
  v = V(:);
  x1 = homography(1, 1)*u + homography(1, 2)*v + homography(1, 3);
  y1 = homography(2, 1)*u + homography(2, 2)*v + homography(2, 3);
  w1 = 1./(homography(3, 1)*u + homography(3, 2)*v + homography(3, 3));
  U(:) = x1 .* w1;
  V(:) = y1 .* w1;

  % Compute interpolation and set NaN to 0 (black)
  warped_img(nrows, ncols, 3) = 1;
  warped_img = zeros(nrows, ncols, 3);
  warped_img(1:nrows,1:ncols,1) = interp2(img_2(:,:,1), U, V, 'cubic');
  warped_img(1:nrows,1:ncols,2) = interp2(img_2(:,:,2), U, V, 'cubic');
  warped_img(1:nrows,1:ncols,3) = interp2(img_2(:,:,3), U, V, 'cubic');
  warped_img(isnan(warped_img)) = 0;
