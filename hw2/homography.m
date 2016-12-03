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

function H = homography(num, x_1, y_1, x_2, y_2)
  A = zeros(2*num, 8);
  % Solve linear equation
  for i = 1 : num
    A(2*i-1,:) = [x_1(i), y_1(i), 1, 0, 0, 0, -x_1(i)*x_2(i) -y_1(i)*x_2(i)];
    A(2*i  ,:) = [0, 0, 0, x_1(i), y_1(i), 1, -x_1(i)*y_2(i) -y_1(i)*y_2(i)];
  end

  b = zeros(2*num, 1);
  for i = 1 : num
    b(2*i-1) = x_2(i);
    b(2*i)   = y_2(i);
  end

  % Compute Ax = b
  h = A\b;
  h(9) = 1
  H = transpose(reshape(h, [3, 3]))