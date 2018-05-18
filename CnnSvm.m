%% Read me
%This project use CNN to get the feature,and use one-class svm to classify
%This project have three CNN-Model(alexnet,vgg16,vgg19)

%% Applying Deeplearning to Anomaly Detection for manufacturing product
% This is the way to detect feature outlier with AlexNet and 1-class SVM kernel method. 
clear; close all; imtool close all; clc;rng('default')

%% Read Pre-trained Convolutional Neural Network (CNN) 
convnet = alexnet() %readAndPreproc size is 227*227,
%convnet = vgg16()  %readAndPreproc size is 224*224,
%convnet = vgg19()  %readAndPreproc size is 224*224,

%convnet = googlenet()   %DAGNetwork
%convnet = resnet50()    %DAGNetwork
%% show layers
convnet.Layers % show layer

%% open folder including training images
rootFolder = pwd;
categ = {'trainingimage'};
%winopen('trainingimage')
%winopen('testimage')
%% use imageDatastore object for dealing with huge amount of image.
imds = imageDatastore(fullfile(rootFolder, categ), 'LabelSource', 'foldernames') 
imds.ReadFcn = @readAndPreproc; % set function to resize image to 227*227*3.
tbl = countEachLabel(imds) % Show the number of training image

%% Run AlexNet to get the feature data at the fc7 layer(vgg16,vgg19 fLayer is fc8)
fLayer = 'fc7'; %AlexNet 
%fLayer = 'fc8'; %VGG16 VGG19
%fLayer = 'fc1000';

trainingFeatures = activations(convnet, imds, fLayer, ...
             'MiniBatchSize', 32, 'OutputAs', 'columns');      % run the network with images and get the feature data at the defined layer

%% train a 1-class SVM with the feature data 
W = ones(size(trainingFeatures', 1), 1); 
d = fitcsvm(trainingFeatures', W, 'KernelScale', 'auto', 'Standardize', false, 'OutlierFraction', 0.04,'KernelFunction','gaussian');

%% Detect 4 abnormal images from test image set 
categ2 = {'testimage'};
% Read 100 images as a test set
imds2 = imageDatastore(fullfile(rootFolder, categ2), 'LabelSource', 'foldernames','IncludeSubfolders',true)
imds2.ReadFcn = @readAndPreproc;
tic % start timer
testFeatures = activations(convnet, imds2, fLayer, ...
             'MiniBatchSize', 32, 'OutputAs', 'columns');  % Execute Alexnet and get data at the fc7 layer
                  
[~, score] = predict(d, testFeatures'); % predict score with trained SVM 
[score_sorted, idx] = sort(score); % sort by score (is score is small (like negative), the image can be abnormal)
idx(1:25)  % the indices of Top 25 abnormal images
toc  % Stop time and show the calculation time
%% Get Detector Num 
hh = [score_sorted,idx];
DetectorNum = [];
C = -1;

for i=1:length(score_sorted)
    if (hh(i,1) <= C)
        DetectorNum(i) = hh(i,2);
    end
end
%% show the sorted images side-by-side
im = readall(imds2);
im = im(idx); % sort images by score in ascending order

LD = length(DetectorNum)
LDImage = im(DetectorNum)
% figure();
% imshow(im{1})
% figure();
% imshow(im{2})
% figure();
% imshow(im{3})
% figure();
% imshow(im{4})

sz = size(im{1});
% Insert rectangle on images people defined as anomaly
for i=1:numel(idx)
     if idx(i) <= LD
        im{i} = insertShape(uint8(im{i}),'rectangle',[1 1 sz(1) sz(2)],'LineWidth' ,10);
     end
end
I = cat(4, im{1:100}); 
figure,montage(I, 'Size', [10 10]) % show 10*10 images in a figure
% The score of images in the first row are low. (anomalousness is high) 
% the 1-4 lowest score images have rectangle yellow frame.
% This means that prediction by classifier is same as the correct answer people define.
score(idx); %

%% Use t-SNE for visualization
rng default % 
testLabels = imds2.Labels; % Use label for visualization 
% Use t-SNE to visualize 4096 dimension data bidimensionally
Y = tsne(testFeatures','Algorithm','exact','NumPCAComponents',50,'Perplexity',45);
figure
gscatter(Y(:,1),Y(:,2),testLabels)
title('Default Figure')
% feature plots of abnormal image are located far from center of whole distribution
% classifier detects these outliers

%% Copyright 2017 The MathWorks, Inc.