function [Xtrain, trainLabels, Xtest, testLabels, imageSize, testFiles] = ...
    load_face_dataset(datasetType, datasetDir, subjectIds, trainPerSubject)




Xtrain = [];
Xtest = [];
trainLabels = [];
testLabels = [];
testFiles = {};
imageSize = [];

for subject = subjectIds(:)'
    switch lower(datasetType)
        case 'orl'
            subjectDir = fullfile(datasetDir, sprintf('s%d', subject));
        case 'yale'
            subjectDir = fullfile(datasetDir, sprintf('yaleB%02d', subject));
        otherwise
            error('Unsupported dataset type: %s', datasetType);
    end
    assert(isfolder(subjectDir), 'Missing subject directory: %s', subjectDir);

    files = dir(fullfile(subjectDir, '*.pgm'));
    assert(~isempty(files), 'No PGM files found in: %s', subjectDir);
    files = local_sort_files(files, datasetType);

    faces = {};
    filenames = {};
    for k = 1:numel(files)
        filename = fullfile(files(k).folder, files(k).name);
        try
            image = double(imread(filename));
            if isempty(imageSize)
                imageSize = size(image);
            end
            if ~isequal(size(image), imageSize)
                warning('Skipping different-sized image: %s', filename);
                continue;
            end
            faces{end + 1} = image(:);
            filenames{end + 1} = filename;
        catch exception
            warning('Skipping unreadable image: %s (%s)', filename, exception.message);
        end
    end

    assert(numel(faces) > trainPerSubject, ...
        ['Subject %d has only %d readable images; need more than %d ' ...
         'for train/test evaluation.'], subject, numel(faces), trainPerSubject);

    for k = 1:trainPerSubject
        Xtrain(:, end + 1) = faces{k};
        trainLabels(end + 1, 1) = subject;
    end
    for k = trainPerSubject + 1:numel(faces)
        Xtest(:, end + 1) = faces{k};
        testLabels(end + 1, 1) = subject;
        testFiles{end + 1, 1} = filenames{k};
    end
end
end

function files = local_sort_files(files, datasetType)
if strcmpi(datasetType, 'orl')
    imageNumbers = zeros(numel(files), 1);
    for k = 1:numel(files)
        token = regexp(files(k).name, '^\d+', 'match', 'once');
        imageNumbers(k) = str2double(token);
    end
    [~, order] = sort(imageNumbers);
else
    [~, order] = sort({files.name});
end
files = files(order);
end
