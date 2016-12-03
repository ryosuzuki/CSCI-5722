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

clear all;close all;clc;



%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task. 1
% Take four points per each image
% which are a minium number of points for Homography transformation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

num = 4
% Read example iamges
img_1 = imread('images/uttower1.JPG');
img_2 = imread('images/uttower2.JPG');

% Get 4 points
imshow(img_1);
[x_1, y_1] = ginput(num);
imshow(img_2);
[x_2, y_2] = ginput(num);

% For debugging
% x_1 = [7, 448, 324, 4]
% y_1 = [113, 297, 503, 493]
% x_2 = [480, 891, 786, 463]
% y_2 = [176, 318, 537, 533]



%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task. 2
% Get Homography matrix H
% that converts [x, y, 1] into [x', y', z]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calcurate Homography matrix
H = homography(num, x_1, y_1, x_2, y_2);

disp('Check Holography matrix')
test_1 = [x_1(1); y_1(1); 1];
test_2 = [x_2(1); y_2(1); 1];
result = H * test_1;
lambda = 1 / result(3);
H * test_1 * lambda
disp('should be')
test_2



%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task. 3
% Warping between two images
% by using Homography matrix H
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

invH = inv(H);

disp('Check inversed Holography matrix')
result = invH * test_2;
lambda = 1 / result(3);
invH * test_2 * lambda
disp('should be')
test_1

img_1 = im2double(img_1);
img_2 = im2double(img_2);

% Get warped image
warped_img = warp(img_2, invH);
imshow(warped_img);
imwrite(warped_img, 'warped_1.jpg');



%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task. 4
% Blend the two images
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get mosaic image
mosaic_img = mosaic(img_2, img_1, invH);
imshow(mosaic_img);



%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task. 5
% Save the blend image
% and test my onw images
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

imwrite(mosaic_img, 'mosaic_1.jpg');

img_1 = imread('images/mountain1.JPG');
img_2 = imread('images/mountain2.JPG');
imshow(img_1);
[x_1, y_1] = ginput(num);
imshow(img_2);
[x_2, y_2] = ginput(num);

H = homography(num, x_1, y_1, x_2, y_2);
invH = inv(H);
img_1 = im2double(img_1);
img_2 = im2double(img_2);

warped_img = warp(img_2, invH);
imshow(warped_img);
imwrite(warped_img, 'warped_2.jpg');

mosaic_img = mosaic(img_2, img_1, invH);
imshow(mosaic_img);
imwrite(mosaic_img, 'mosaic_2.jpg');




%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Optional
% Use RANSAC function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%


img_1 = imread('images/uttower1.JPG');
img_2 = imread('images/uttower2.JPG');
img_1 = im2double(img_1);
img_2 = im2double(img_2);

x_1 = [7, 448, 324, 4]
y_1 = [113, 297, 503, 493]
x_2 = [480, 891, 786, 463]
y_2 = [176, 318, 537, 533]

[best_H, best_inlier_count] = ransac(num, x_1, x_2, y_1, y_2);
inv_best_H = inv(best_H);

warped_img = warp(img_2, inv_best_H);
imshow(warped_img);
imwrite(warped_img, 'warped_3.jpg');

mosaic_img = mosaic(img_2, img_1, inv_best_H);
imshow(mosaic_img);
imwrite(mosaic_img, 'mosaic_3.jpg');


