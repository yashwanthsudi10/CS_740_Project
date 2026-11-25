





clear; clc; close all;
rng(42);
projectDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(projectDir, 'results_complete');
if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

configs = [ ...
    struct('name', 'ORL', 'type', 'orl', 'pathNames', {{'orl', 'ORL'}}, ...
           'subjectIds', 1:32, 'trainingCounts', 2:6, 'dimensionTrainCount', 6), ...
    struct('name', 'Yale', 'type', 'yale', ...
           'pathNames', {{'CroppedYale', 'croppedYale', 'Yale'}}, ...
           'subjectIds', setdiff(1:39, 14), ...
           'trainingCounts', [10 20 30 40], 'dimensionTrainCount', 40) ...
    ];

allSummary = table();
allDimensions = table();

for datasetIndex = 1:numel(configs)
    config = configs(datasetIndex);
    datasetDir = local_find_dataset(projectDir, config.pathNames);
    assert(~isempty(datasetDir), 'Could not find the %s dataset folder.', config.name);
    fprintf('\n========== %s ==========\n', config.name);

    for trainPerSubject = config.trainingCounts
        fprintf('Running %s with %d training images per subject...\n', ...
            config.name, trainPerSubject);
        [Xtrain, trainLabels, Xtest, testLabels] = load_face_dataset( ...
            config.type, datasetDir, config.subjectIds, trainPerSubject);
        [summary, ~, ~] = evaluate_face_experiment(config.name, trainPerSubject, ...
            Xtrain, trainLabels, Xtest, testLabels);
        allSummary = [allSummary; summary];
    end


    [Xtrain, trainLabels, Xtest, testLabels] = load_face_dataset( ...
        config.type, datasetDir, config.subjectIds, config.dimensionTrainCount);
    dimensionResults = local_dimension_study( ...
        config.name, Xtrain, trainLabels, Xtest, testLabels);
    allDimensions = [allDimensions; dimensionResults];


    local_save_demo(config.name, Xtrain, trainLabels, Xtest, testLabels, ...
        resultsDir);
end

writetable(allSummary, fullfile(resultsDir, 'summary_by_training_size.csv'));
writetable(allDimensions, fullfile(resultsDir, 'accuracy_by_dimension.csv'));
local_plot_training_size(allSummary, resultsDir);
local_plot_dimensions(allDimensions, resultsDir);
save(fullfile(resultsDir, 'complete_project_results.mat'), ...
    'allSummary', 'allDimensions', 'configs');

fprintf('\nComplete results saved to: %s\n', resultsDir);

function datasetDir = local_find_dataset(projectDir, pathNames)
datasetDir = '';
for candidateName = pathNames
    possibleLocations = { ...
        fullfile(projectDir, candidateName{1}), ...
        fullfile(projectDir, '..', candidateName{1}) ...
        };
    for locationIndex = 1:numel(possibleLocations)
        candidate = possibleLocations{locationIndex};
        if isfolder(candidate)
            datasetDir = candidate;
            return;
        end
    end
end
end

function dimensionResults = local_dimension_study( ...
    datasetName, Xtrain, trainLabels, Xtest, testLabels)
numClasses = numel(unique(trainLabels));
maxPca = min(size(Xtrain, 2) - numClasses, size(Xtrain, 2) - 1);
maxFisher = min(numClasses - 1, maxPca);
eigenModel = train_eigenfaces(Xtrain, maxPca);
fisherModel = train_fisherfaces(Xtrain, trainLabels);
eigenDims = unique(min([5 10 20 30 50 75 100 150 maxPca], maxPca));
fisherDims = unique(min([2 5 10 15 20 25 30 maxFisher], maxFisher));

dimensionResults = table();
for metric = ["Euclidean", "Correlation"]
    for dimensions = eigenDims
        prediction = predict_nearest_neighbor(eigenModel, Xtrain, trainLabels, ...
            Xtest, dimensions, lower(metric));
        row = table(string(datasetName), "Eigenfaces", metric, dimensions, ...
            100 * mean(prediction == testLabels), ...
            'VariableNames', {'Dataset', 'Method', 'Metric', ...
            'Dimensions', 'AccuracyPercent'});
        dimensionResults = [dimensionResults; row];
    end
    for dimensions = fisherDims
        prediction = predict_nearest_neighbor(fisherModel, Xtrain, trainLabels, ...
            Xtest, dimensions, lower(metric));
        row = table(string(datasetName), "Fisherfaces", metric, dimensions, ...
            100 * mean(prediction == testLabels), ...
            'VariableNames', {'Dataset', 'Method', 'Metric', ...
            'Dimensions', 'AccuracyPercent'});
        dimensionResults = [dimensionResults; row];
    end
end

end

function local_save_demo(datasetName, Xtrain, trainLabels, Xtest, testLabels, ...
    resultsDir)


if strcmpi(datasetName, 'ORL')
    imageSize = [112 92];
else
    imageSize = [192 168];
end
model = train_fisherfaces(Xtrain, trainLabels);
[label, index] = predict_nearest_neighbor(model, Xtrain, trainLabels, ...
    Xtest(:, 1), size(model.basis, 2), 'correlation');
figure('Color', 'w', 'Position', [100 100 720 350]);
subplot(1, 2, 1);
imagesc(reshape(Xtest(:, 1), imageSize)); axis image off; colormap gray;
title({sprintf('%s test face', datasetName), ...
    sprintf('True: subject %d', testLabels(1))});
subplot(1, 2, 2);
imagesc(reshape(Xtrain(:, index), imageSize)); axis image off;
title({'Nearest Fisherface match', ...
    sprintf('Predicted: subject %d', label)});
exportgraphics(gcf, fullfile(resultsDir, ...
    sprintf('%s_fisherfaces_demo.png', lower(datasetName))), 'Resolution', 200);
close(gcf);
end

function local_plot_training_size(summary, resultsDir)
figure('Color', 'w', 'Position', [100 100 1000 650]);
datasets = unique(summary.Dataset);
for k = 1:numel(datasets)
    subplot(1, numel(datasets), k);
    subset = summary(summary.Dataset == datasets(k), :);
    groups = unique(strcat(subset.Method, " - ", subset.Metric));
    for g = 1:numel(groups)
        selection = strcat(subset.Method, " - ", subset.Metric) == groups(g);
        rows = subset(selection, :);
        plot(rows.TrainPerSubject, rows.AccuracyPercent, '-o', ...
            'LineWidth', 1.6, 'DisplayName', groups(g));
        hold on;
    end
    grid on; xlabel('Training images per person'); ylabel('Accuracy (%)');
    title(sprintf('%s: training-size comparison', datasets(k)));
    legend('Location', 'best');
end
exportgraphics(gcf, fullfile(resultsDir, 'accuracy_vs_training_images.png'), ...
    'Resolution', 200);
close(gcf);
end

function local_plot_dimensions(results, resultsDir)
figure('Color', 'w', 'Position', [100 100 1000 650]);
datasets = unique(results.Dataset);
for k = 1:numel(datasets)
    subplot(1, numel(datasets), k);
    subset = results(results.Dataset == datasets(k), :);
    groups = unique(strcat(subset.Method, " - ", subset.Metric));
    for g = 1:numel(groups)
        selection = strcat(subset.Method, " - ", subset.Metric) == groups(g);
        rows = subset(selection, :);
        plot(rows.Dimensions, rows.AccuracyPercent, '-o', ...
            'LineWidth', 1.6, 'DisplayName', groups(g));
        hold on;
    end
    grid on; xlabel('Retained dimensions'); ylabel('Accuracy (%)');
    title(sprintf('%s: dimensionality comparison', datasets(k)));
    legend('Location', 'best');
end
exportgraphics(gcf, fullfile(resultsDir, 'accuracy_vs_dimensions.png'), ...
    'Resolution', 200);
close(gcf);
end
