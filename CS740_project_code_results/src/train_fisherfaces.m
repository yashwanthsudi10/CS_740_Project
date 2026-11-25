function model = train_fisherfaces(Xtrain, labels)


meanFace = mean(Xtrain, 2);
A = Xtrain - meanFace;
classes = unique(labels)';
numClasses = numel(classes);
numSamples = size(Xtrain, 2);


pcaDims = min(numSamples - numClasses, size(A, 2) - 1);
L = (A' * A) / numSamples;
[V, D] = eig(L, 'vector');
[eigenvalues, order] = sort(real(D), 'descend');
V = real(V(:, order));
valid = eigenvalues > max(eigenvalues) * 1e-12;
eigenvalues = eigenvalues(valid);
V = V(:, valid);

pcaDims = min(pcaDims, size(V, 2));
Wpca = A * V(:, 1:pcaDims) ./ sqrt(numSamples * eigenvalues(1:pcaDims)');
Wpca = Wpca ./ vecnorm(Wpca);
Z = Wpca' * A;


overallMean = mean(Z, 2);
Sw = zeros(pcaDims, pcaDims);
Sb = zeros(pcaDims, pcaDims);
for classId = classes
    Zclass = Z(:, labels == classId);
    classMean = mean(Zclass, 2);
    centeredClass = Zclass - classMean;
    Sw = Sw + centeredClass * centeredClass';
    count = size(Zclass, 2);
    delta = classMean - overallMean;
    Sb = Sb + count * (delta * delta');
end



regularizer = 1e-8 * trace(Sw) / max(1, pcaDims);
[Wlda, Dlda] = eig(Sb, Sw + regularizer * eye(pcaDims), 'vector');
[fisherValues, order] = sort(real(Dlda), 'descend');
Wlda = real(Wlda(:, order));

fisherDims = min(numClasses - 1, size(Wlda, 2));
Wfisher = Wpca * Wlda(:, 1:fisherDims);
Wfisher = Wfisher ./ vecnorm(Wfisher);

model.meanFace = meanFace;
model.basis = Wfisher;
model.pcaBasis = Wpca;
model.ldaBasis = Wlda(:, 1:fisherDims);
model.pcaDimensions = pcaDims;
model.fisherValues = fisherValues(1:fisherDims);
model.withinClassScatter = trace(Sw);
model.betweenClassScatter = trace(Sb);
model.ldaSeparationRatio = trace(Sb) / max(trace(Sw), eps);
model.method = 'Fisherfaces';
end
