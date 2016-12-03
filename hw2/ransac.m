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

function [best_H, max_inl] = ransac(num, x_1, y_1, x_2, y_2)

  A = zeros(2 * num, 8);
  b = zeros(2 * num, 1);
  best_H = zeros(3,3);
  max_inl = -Inf;

  % TODO: Replace with RANSAC function
  best_H = homography(num, x_1, y_1, x_2, y_2);


