function [Xtrain, trainLabels, Xtest, testLabels, imageSize, testFiles] = ...
    load_orl_dataset(datasetDir, numSubjects, trainImageNumbers, testImageNumbers)


Xtrain = [];
Xtest = [];
trainLabels = [];
testLabels = [];
testFiles = {};
imageSize = [];

for subject = 1:numSubjects
    subjectDir = fullfile(datasetDir, sprintf('s%d', subject));
    assert(isfolder(subjectDir), 'Missing subject directory: %s', subjectDir);

    for imageNumber = trainImageNumbers
        filename = fullfile(subjectDir, sprintf('%d.pgm', imageNumber));
        image = double(imread(filename));
        if isempty(imageSize)
            imageSize = size(image);
        end
        assert(isequal(size(image), imageSize), ...
            'All images must have identical dimensions.');
        Xtrain(:, end + 1) = image(:);
        trainLabels(end + 1, 1) = subject;
    end

    for imageNumber = testImageNumbers
        filename = fullfile(subjectDir, sprintf('%d.pgm', imageNumber));
        image = double(imread(filename));
        assert(isequal(size(image), imageSize), ...
            'All images must have identical dimensions.');
        Xtest(:, end + 1) = image(:);
        testLabels(end + 1, 1) = subject;
        testFiles{end + 1, 1} = filename;
    end
end
end
