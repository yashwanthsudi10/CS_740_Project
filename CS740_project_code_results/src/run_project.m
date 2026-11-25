



clear; clc; close all;
rng(42);

projectDir = fileparts(mfilename('fullpath'));
datasetCandidates = {fullfile(projectDir, '..', 'orl'), ...
                     fullfile(projectDir, '..', 'ORL'), ...
                     fullfile(projectDir, 'orl'), ...
                     fullfile(projectDir, 'ORL')};
datasetDir = '';
for k = 1:numel(datasetCandidates)
    if isfolder(datasetCandidates{k})
        datasetDir = datasetCandidates{k};
        break;
    end
end
assert(~isempty(datasetDir), ...
    ['ORL dataset folder not found. Put this project next to orl/ ' ...
     '(or ORL/) and run again.']);

numSubjects = 32;
[Xtrain, trainLabels, Xtest, testLabels, imageSize, testFiles] = ...
    load_orl_dataset(datasetDir, numSubjects, 1:6, 7:10);

numClasses = numel(unique(trainLabels));
numTrain = size(Xtrain, 2);
maxPca = min(numTrain - numClasses, size(Xtrain, 2) - 1);
maxFisher = numClasses - 1;

fprintf('Training faces: %d | Test faces: %d | Classes: %d\n', ...
    size(Xtrain, 2), size(Xtest, 2), numClasses);
fprintf('PCA-before-LDA dimension: %d | Maximum Fisherface dimensions: %d\n', ...
    maxPca, maxFisher);


eigenModel = train_eigenfaces(Xtrain, maxPca);
fisherModel = train_fisherfaces(Xtrain, trainLabels);


eigenDims = unique(min([5 10 15 20 30 50 75 100 maxPca], maxPca));
fisherDims = unique(min([2 5 10 15 20 25 maxFisher], maxFisher));

eigenEuclideanAccuracy = zeros(numel(eigenDims), 1);
eigenCorrelationAccuracy = zeros(numel(eigenDims), 1);
for k = 1:numel(eigenDims)
    predicted = predict_nearest_neighbor(eigenModel, Xtrain, trainLabels, ...
        Xtest, eigenDims(k), 'euclidean');
    eigenEuclideanAccuracy(k) = mean(predicted == testLabels) * 100;
    predicted = predict_nearest_neighbor(eigenModel, Xtrain, trainLabels, ...
        Xtest, eigenDims(k), 'correlation');
    eigenCorrelationAccuracy(k) = mean(predicted == testLabels) * 100;
end

fisherEuclideanAccuracy = zeros(numel(fisherDims), 1);
fisherCorrelationAccuracy = zeros(numel(fisherDims), 1);
for k = 1:numel(fisherDims)
    predicted = predict_nearest_neighbor(fisherModel, Xtrain, trainLabels, ...
        Xtest, fisherDims(k), 'euclidean');
    fisherEuclideanAccuracy(k) = mean(predicted == testLabels) * 100;
    predicted = predict_nearest_neighbor(fisherModel, Xtrain, trainLabels, ...
        Xtest, fisherDims(k), 'correlation');
    fisherCorrelationAccuracy(k) = mean(predicted == testLabels) * 100;
end

resultsDir = fullfile(projectDir, 'results');
if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

T = table([repmat("Eigenfaces", 2 * numel(eigenDims), 1); ...
           repmat("Fisherfaces", 2 * numel(fisherDims), 1)], ...
          [repmat("Euclidean", numel(eigenDims), 1); ...
           repmat("Correlation", numel(eigenDims), 1); ...
           repmat("Euclidean", numel(fisherDims), 1); ...
           repmat("Correlation", numel(fisherDims), 1)], ...
          [eigenDims(:); eigenDims(:); fisherDims(:); fisherDims(:)], ...
          [eigenEuclideanAccuracy; eigenCorrelationAccuracy; ...
           fisherEuclideanAccuracy; fisherCorrelationAccuracy], ...
          'VariableNames', {'Method', 'Metric', 'Dimensions', 'AccuracyPercent'});
writetable(T, fullfile(resultsDir, 'accuracy_comparison.csv'));

figure('Color', 'w', 'Position', [100 100 850 520]);
plot(eigenDims, eigenEuclideanAccuracy, '-o', 'LineWidth', 1.8, ...
    'DisplayName', 'Eigenfaces - Euclidean');
hold on;
plot(eigenDims, eigenCorrelationAccuracy, '--o', 'LineWidth', 1.8, ...
    'DisplayName', 'Eigenfaces - Correlation');
plot(fisherDims, fisherEuclideanAccuracy, '-s', 'LineWidth', 1.8, ...
    'DisplayName', 'Fisherfaces - Euclidean');
plot(fisherDims, fisherCorrelationAccuracy, '--s', 'LineWidth', 1.8, ...
    'DisplayName', 'Fisherfaces - Correlation');
grid on;
xlabel('Number of retained dimensions');
ylabel('Recognition accuracy (%)');
title('ORL Face Recognition: Eigenfaces vs. Fisherfaces');
legend('Location', 'best');
exportgraphics(gcf, fullfile(resultsDir, 'accuracy_comparison.png'), ...
    'Resolution', 200);


[demoLabel, demoIndex, demoDistance] = predict_nearest_neighbor( ...
    fisherModel, Xtrain, trainLabels, Xtest(:, 1), maxFisher, 'correlation');
fprintf('\nDemo image: %s\n', testFiles{1});
fprintf('True subject: s%d | Fisherfaces correlation prediction: s%d | 1-r: %.4f\n', ...
    testLabels(1), demoLabel, demoDistance);

figure('Color', 'w', 'Position', [100 100 720 350]);
subplot(1, 2, 1);
imagesc(reshape(Xtest(:, 1), imageSize)); axis image off; colormap gray;
title(sprintf('Test image\nTrue: s%d', testLabels(1)));
subplot(1, 2, 2);
imagesc(reshape(Xtrain(:, demoIndex), imageSize)); axis image off;
title(sprintf('Nearest Fisherface match\nPredicted: s%d', demoLabel));
exportgraphics(gcf, fullfile(resultsDir, 'demo_prediction.png'), ...
    'Resolution', 200);

save(fullfile(resultsDir, 'experiment_results.mat'), ...
    'eigenModel', 'fisherModel', 'eigenDims', 'fisherDims', ...
    'eigenEuclideanAccuracy', 'eigenCorrelationAccuracy', ...
    'fisherEuclideanAccuracy', 'fisherCorrelationAccuracy', ...
    'T', 'imageSize');

fprintf('\nSaved results to: %s\n', resultsDir);
fprintf('Best Eigenfaces (Euclidean): %.2f%%\n', max(eigenEuclideanAccuracy));
fprintf('Best Eigenfaces (Correlation): %.2f%%\n', max(eigenCorrelationAccuracy));
fprintf('Best Fisherfaces (Euclidean): %.2f%%\n', max(fisherEuclideanAccuracy));
fprintf('Best Fisherfaces (Correlation): %.2f%%\n', max(fisherCorrelationAccuracy));
