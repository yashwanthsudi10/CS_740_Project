function model = train_eigenfaces(Xtrain, requestedDims)


meanFace = mean(Xtrain, 2);
A = Xtrain - meanFace;
numSamples = size(A, 2);


L = (A' * A) / numSamples;
[V, D] = eig(L, 'vector');
[eigenvalues, order] = sort(real(D), 'descend');
V = real(V(:, order));

valid = eigenvalues > max(eigenvalues) * 1e-12;
eigenvalues = eigenvalues(valid);
V = V(:, valid);
U = A * V ./ sqrt(numSamples * eigenvalues');
U = U ./ vecnorm(U);

requestedDims = min(requestedDims, size(U, 2));
model.meanFace = meanFace;
model.basis = U(:, 1:requestedDims);
model.eigenvalues = eigenvalues(1:requestedDims);
model.method = 'Eigenfaces';
end
