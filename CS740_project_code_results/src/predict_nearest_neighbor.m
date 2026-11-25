function [predictedLabels, nearestIndices, nearestDistances] = ...
    predict_nearest_neighbor(model, Xtrain, trainLabels, Xquery, dimensions, metric)



if nargin < 6
    metric = 'euclidean';
end

dimensions = min(dimensions, size(model.basis, 2));
if strcmpi(metric, 'correlation')
    assert(dimensions >= 2, ...
        'Correlation matching needs at least two retained dimensions.');
end
W = model.basis(:, 1:dimensions);
trainFeatures = W' * (Xtrain - model.meanFace);
queryFeatures = W' * (Xquery - model.meanFace);

numQueries = size(Xquery, 2);
predictedLabels = zeros(numQueries, 1);
nearestIndices = zeros(numQueries, 1);
nearestDistances = zeros(numQueries, 1);

for q = 1:numQueries
    switch lower(metric)
        case 'euclidean'
            distances = sum((trainFeatures - queryFeatures(:, q)).^2, 1);
        case 'correlation'


            trainCentered = trainFeatures - mean(trainFeatures, 1);
            queryCentered = queryFeatures(:, q) - mean(queryFeatures(:, q));
            queryNorm = norm(queryCentered);
            trainNorms = sqrt(sum(trainCentered.^2, 1));
            denominators = trainNorms(:) * queryNorm;
            denominators(denominators < eps) = eps;
            correlations = (trainCentered' * queryCentered) ./ denominators;
            distances = (1 - correlations)';
        otherwise
            error('Unknown metric: %s. Use euclidean or correlation.', metric);
    end
    [minimumDistance, index] = min(distances);
    predictedLabels(q) = trainLabels(index);
    nearestIndices(q) = index;
    if strcmpi(metric, 'euclidean')
        nearestDistances(q) = sqrt(minimumDistance);
    else
        nearestDistances(q) = minimumDistance;
    end
end
end
