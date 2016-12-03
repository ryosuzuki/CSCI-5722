% This script creates a menu driven application

%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ryo Suzuki
% rysu7393
% 105790212
% ryo.suzuki@colorado.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;

% Display a menu and get a choice
choice = menu('Choose an option', 'Exit Program', 'Load Image', 'Display Image', 'Mean Filter', 'Gaussian Filter', 'Scale Nearest', 'Scale Bilnear', 'Fun Filter', 'Mean Filter');


% Choice 1 is to exit the program
while choice ~= 1
  switch choice
    case 0
      disp('Error - please choose one of the options.')
      % Display a menu and get a choice
      choice = menu('Choose an option', 'Exit Program', 'Load Image', 'Display Image', 'Mean Filter', 'Gaussian Filter', 'Scale Nearest', 'Scale Bilnear', 'Fun Filter', 'Mean Filter');
    case 2
      % Load an image
      image_choice = menu('Choose an image', 'lena1', 'mandril1', 'sully', 'yoda', 'shrek');
      switch image_choice
        case 1
          filename = 'lena1.jpg';
        case 2
          filename = 'mandrill1.jpg';
        case 3
          filename = 'sully.bmp'
        case 4
          filename = 'yoda.bmp';
        case 5
          filename = 'shrek.bmp'
        end
      current_img = imread(filename);
    case 3
      % Display image
      figure
      imagesc(current_img);
      if size(current_img,3) == 1
        colormap gray
      end
    case 4
      % Mean Filter
      res = inputdlg('Choose a kernel size');
      kernel_size = str2num(res{1});
      new_img = mean_filter(current_img, kernel_size);
      display_image(current_img, new_img, 'mean_filter');
    case 5
      % Gaussian Filter
      res = inputdlg('Input a positive value for sigma');
      sigma = str2double(res{1});
      new_img = gaussian_filter(current_img, sigma);
      display_image(current_img, new_img, 'gaussian_filter');
    case 6
      % Scale Nearest
      res = inputdlg('Input a positive value for factor');
      factor = str2double(res{1});
      new_img = scale_nearest(current_img, factor);
      display_image(current_img, new_img, 'scale_nearest');
    case 7
      % Scale Bilinear
      res = inputdlg('Input a positive value for factor');
      factor = str2double(res{1});
      new_img = scale_bilinear(current_img, factor);
      display_image(current_img, new_img, 'scale_bilinear');
    case 8
      % Fun Filter
      new_img = fun_filter(current_img);
      display_image(current_img, new_img, 'fun_filter');
    end
  % Display menu again and get user's choice
  choice = menu('Choose an option', 'Exit Program', 'Load Image', 'Display Image', 'Mean Filter', 'Gaussian Filter', 'Scale Nearest', 'Scale Bilnear', 'Fun Filter', 'Mean Filter');
end