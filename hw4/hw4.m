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


X = 'HORSEBACK'
Y = 'SNOWFLAKE'

[c, b, dist, str] = lcs(X, Y)

%% Depth Estimation From Stereo Video
% This example shows how to detect people in video taken with a calibrated
% stereo camera and determine their distances from the camera.
%
%   Copyright 2013-2014 The MathWorks, Inc.

%% Load the Parameters of the Stereo Camera
% Load the |stereoParameters| object, which is the result of calibrating
% the camera using either the |stereoCameraCalibrator| app or the
% |estimateCameraParameters| function.

% Load the stereoParameters object.
load('handshakeStereoParams.mat');

% Visualize camera extrinsics.
% showExtrinsics(stereoParams);

%% Create Video File Readers and the Video Player
% Create System Objects for reading and displaying the video
videoFileLeft = 'handshake_left.avi';
videoFileRight = 'handshake_right.avi';

readerLeft = vision.VideoFileReader(videoFileLeft, 'VideoOutputDataType', 'uint8');
readerRight = vision.VideoFileReader(videoFileRight, 'VideoOutputDataType', 'uint8');
player = vision.DeployableVideoPlayer('Location', [20, 400]);

%% Read and Rectify Video Frames
% The frames from the left and the right cameras must be rectified in order
% to compute disparity and reconstruct the 3-D scene. Rectified images
% have horizontal epipolar lines, and are row-aligned. This simplifies
% the computation of disparity by reducing the search space for matching
% points to one dimension.  Rectified images can also be combined into an
% anaglyph, which can be viewed using the stereo red-cyan glasses to see
% the 3-D effect.
frameLeft = readerLeft.step();
frameRight = readerRight.step();

[frameLeftRect, frameRightRect] = ...
    rectifyStereoImages(frameLeft, frameRight, stereoParams);

% figure;
% imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
% title('Rectified Video Frames');

%% Compute Disparity
% In rectified stereo images any pair of corresponding points are located
% on the same pixel row. For each pixel in the left image compute the
% distance to the corresponding pixel in the right image. This distance is
% called the disparity, and it is proportional to the distance of the
% corresponding world point from the camera.

imageLeft = frameLeftRect;
imageRight = frameRightRect;

imageLeft = imread('images/cone-left.png');
imageRight = imread('images/cone-right.png');

imageLeft = imread('images/teddy-left.png');
imageRight = imread('images/teddy-right.png');

frameLeftGray = rgb2gray(imageLeft);
frameRightGray = rgb2gray(imageRight);
disparityRange = [0, 64];


disp('Start computing the disparity map');
[disparityMap, D, b] = stereoDP(frameLeftGray, frameRightGray, 64, 0.01);

disp('Start colorization');
coloredDisparityMap = displayDmap(disparityMap);

disp('Finish calculation');
figure;
imshow(coloredDisparityMap, disparityRange);
title('Disparity Map');


% %% Reconstruct the 3-D Scene
% % Reconstruct the 3-D world coordinates of points corresponding to each
% % pixel from the disparity map.
% points3D = reconstructScene(disparityMap, stereoParams);

% % Convert to meters and create a pointCloud object
% points3D = points3D ./ 1000;
% ptCloud = pointCloud(points3D, 'Color', frameLeftRect);

% % Create a streaming point cloud viewer
% player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
%     'VerticalAxisDir', 'down');

% % Visualize the point cloud
% view(player3D, ptCloud);


% %% Detect People in the Left Image
% % Use the |vision.PeopleDetector| system object to detect people.

% % Create the people detector object. Limit the minimum object size for
% % speed.
% peopleDetector = vision.PeopleDetector('MinSize', [166 83]);

% % Detect people.
% bboxes = peopleDetector.step(frameLeftGray);

% %% Determine The Distance of Each Person to the Camera
% % Find the 3-D world coordinates of the centroid of each detected person
% % and compute the distance from the centroid to the camera in meters.

% % Find the centroids of detected people.
% centroids = [round(bboxes(:, 1) + bboxes(:, 3) / 2), ...
%     round(bboxes(:, 2) + bboxes(:, 4) / 2)];

% % Find the 3-D world coordinates of the centroids.
% centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
% X = points3D(:, :, 1);
% Y = points3D(:, :, 2);
% Z = points3D(:, :, 3);
% centroids3D = [X(centroidsIdx)'; Y(centroidsIdx)'; Z(centroidsIdx)'];

% % Find the distances from the camera in meters.
% dists = sqrt(sum(centroids3D .^ 2));

% % Display the detected people and their distances.
% labels = cell(1, numel(dists));
% for i = 1:numel(dists)
%     labels{i} = sprintf('%0.2f meters', dists(i));
% end
% figure;
% imshow(insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes, labels));
% title('Detected People');

% %% Process the Rest of the Video
% % Apply the steps described above to detect people and measure their
% % distances to the camera in every frame of the video.

% while ~isDone(readerLeft) && ~isDone(readerRight)
%     % Read the frames.
%     frameLeft = readerLeft.step();
%     frameRight = readerRight.step();

%     % Rectify the frames.
%     [frameLeftRect, frameRightRect] = ...
%         rectifyStereoImages(frameLeft, frameRight, stereoParams);

%     % Convert to grayscale.
%     frameLeftGray  = rgb2gray(frameLeftRect);
%     frameRightGray = rgb2gray(frameRightRect);

%     % Compute disparity.
%     disparityMap = disparity(frameLeftGray, frameRightGray);

%     % Reconstruct 3-D scene.
%     points3D = reconstructScene(disparityMap, stereoParams);
%     points3D = points3D ./ 1000;
%     ptCloud = pointCloud(points3D, 'Color', frameLeftRect);
%     view(player3D, ptCloud);

%     % Detect people.
%     bboxes = peopleDetector.step(frameLeftGray);

%     if ~isempty(bboxes)
%         % Find the centroids of detected people.
%         centroids = [round(bboxes(:, 1) + bboxes(:, 3) / 2), ...
%             round(bboxes(:, 2) + bboxes(:, 4) / 2)];

%         % Find the 3-D world coordinates of the centroids.
%         centroidsIdx = sub2ind(size(disparityMap), centroids(:, 2), centroids(:, 1));
%         X = points3D(:, :, 1);
%         Y = points3D(:, :, 2);
%         Z = points3D(:, :, 3);
%         centroids3D = [X(centroidsIdx), Y(centroidsIdx), Z(centroidsIdx)];

%         % Find the distances from the camera in meters.
%         dists = sqrt(sum(centroids3D .^ 2, 2));

%         % Display the detect people and their distances.
%         labels = cell(1, numel(dists));
%         for i = 1:numel(dists)
%             labels{i} = sprintf('%0.2f meters', dists(i));
%         end
%         dispFrame = insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes,...
%             labels);
%     else
%         dispFrame = frameLeftRect;
%     end

%     % Display the frame.
%     step(player, dispFrame);
% end

% % Clean up.
% reset(readerLeft);
% reset(readerRight);
% release(player);

% %% Summary
% % This example showed how to localize pedestrians in 3-D using a calibrated
% % stereo camera.

% %% References
% % [1] G. Bradski and A. Kaehler, "Learning OpenCV : Computer Vision with
% % the OpenCV Library," O'Reilly, Sebastopol, CA, 2008.
% %
% % [2] Dalal, N. and Triggs, B., Histograms of Oriented Gradients for
% % Human Detection. CVPR 2005.

% displayEndOfDemoMessage(mfilename)


