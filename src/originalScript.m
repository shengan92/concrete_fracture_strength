clc;
clear;
close all;
% Folder containing fracture images
imageFolder = 'fracture_images';
imageFiles = dir(fullfile(imageFolder, '*.jpg'));
% Measured strengths (MPa) – must match image order
strengths = [60.03, 62.57, 61.3,...
            59.87, 62.77, 61.32,...
            65.37, 65.09, 65.23,...
            64.53, 66.37, 65.45,...
            55.14, 57.13, 56.14,...
            58.42, 61.43, 59.93,...
            61.29, 60.37, 60.83,...
            63.32, 64.19, 63.76];
numImages = length(imageFiles);
% Initialize feature matrix
features = zeros(numImages, 2);
features_ave = zeros(numImages/2, 2);
for i = 1:numImages
% Read image
    img = imread(fullfile(imageFolder, imageFiles(i).name));
% Convert to grayscale
    gray = rgb2gray(img);
    gray = im2double(gray);
    gray = (gray - min(gray(:))) / (max(gray(:)) - min(gray(:)));
    gray = adapthisteq(gray);  % contrast limited adaptive histogram equalization
% Resize (optional for consistency)
%gray = imresize(gray, [2500 2500]);
% --- Feature 1: Mean intensity ---
    meanIntensity = mean(gray(:));
% --- Feature 2: Std deviation (texture contrast) ---
    stdIntensity = std(gray(:));
% --- Feature 3: Gradient magnitude (roughness proxy) ---
% [Gmag, ~] = imgradient(gray);
% meanGradient = mean(Gmag(:));
    sigma_values = [1 3 6 12];
    roughness = zeros(1,length(sigma_values));
for s = 1:length(sigma_values)
        blurred = imgaussfilt(gray, sigma_values(s));
        [Gmag,~] = imgradient(blurred);
        roughness(s) = mean(Gmag(:));
end
% --- Feature 4: Edge density ---
% edges = edge(gray, 'Canny');
% edgeDensity = sum(edges(:)) / numel(edges);
    bw = edge(gray, 'Canny');
    skel = bwmorph(bw, 'skel', Inf);
    crackLength = sum(skel(:));
    crackDensity = crackLength / numel(skel);
% --- Feature 5: Porosity estimation via threshold ---
    level = graythresh(gray);
    bw = imbinarize(gray, level);
    porosityFraction = sum(bw(:)) / numel(bw);
% --- Feature 6: GLCM contrast ---
    glcm = graycomatrix(gray, 'Offset', [0 1]);
    stats = graycoprops(glcm);
    textureContrast = stats.Contrast;
% Store features
% features(i,:) = [meanIntensity, stdIntensity, meanGradient, ...
%                  edgeDensity, porosityFraction, textureContrast];
    fineR = mean(roughness(1:2));
    coarseR = mean(roughness(3:4));
%features(i,:) = [fineR, coarseR, crackDensity, porosityFraction];
    features(i,:) = [coarseR, crackDensity];
end
for i = 1:numImages/2
    features_ave(i,:) = (features(2*i-1,:)+features(2*i,:))./2;
end
% Correlation analysis
corrMatrix = corr(features_ave, strengths');
disp('Correlation between features and strength:')
disp(corrMatrix)
% Simple regression model
features_norm = zscore(features_ave);
features_centered = features_ave - mean(features_norm);
mdl = fitlm(features_centered, strengths);
%mdl = fitlm(features_centered, strengths,'quadratic');
%mdl = fitrsvm(features_centered, strengths, 'KernelFunction','gaussian');
disp(mdl)
% cvmdl = crossval(mdl,'KFold',16); %only with fitrsvm
% loss = kfoldLoss(cvmdl);
% R2_cv = 1 - loss/var(strengths);
% Plot predicted vs measured
predicted = predict(mdl, features_centered);
figure;
scatter(strengths, predicted, 100, 'filled')
xlabel('Measured Strength (MPa)')
ylabel('Predicted Strength (MPa)')
title('CoarsR&Porosity')
grid on
