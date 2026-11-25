# CS740: Eigenfaces vs. Fisherfaces

This package contains the MATLAB implementation and the generated experiment results for the Eigenfaces versus Fisherfaces project.

## Contents

- `src/` - MATLAB implementation files. All MATLAB source comments have been removed.
- `results_complete/` - Final ORL and Yale Fisherfaces comparison results.
- `results_pca_baseline/` - Earlier PCA baseline plots, CSV files, and MAT file.

## Run the complete experiment

1. Place the project folder next to the dataset folders:

```text
Assignment_3/
├── CS740_Eigenfaces_Fisherfaces_Submission/
├── ORL/
└── CroppedYale/
```

2. In MATLAB, set the Current Folder to `CS740_Eigenfaces_Fisherfaces_Submission/src`.

3. Run:

```matlab
run_complete_project
```

The script evaluates Eigenfaces and Fisherfaces with Euclidean and correlation matching on ORL and Cropped Yale. New outputs are written to `src/results_complete/`.

## Included final results

`results_complete/` contains:

- `summary_by_training_size.csv`
- `accuracy_by_dimension.csv`
- `accuracy_vs_training_images.png`
- `accuracy_vs_dimensions.png`
- `orl_fisherfaces_demo.png`
- `yale_fisherfaces_demo.png`
- `complete_project_results.mat`

The dataset image folders are intentionally not included because of their size. Download/place the ORL and Cropped Yale datasets next to the package before running the implementation.
