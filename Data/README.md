# Data Folder

This folder contains CSV datasets used in the project.

## Subfolders

- `EyeGazeDataset/` - gaze-region classification data
- `EyesOpenCloseDataset/` - eye open/close state data

## CSV Row/Column Structure

Both datasets use a similar tabular format:

- **One row = one sample (one time step / frame-level measurement)**
- **First column (`Label`) = class label**
- **Remaining columns (`Data_0 ... Data_N`) = numeric sensor features for that sample**

The number of `Data_*` columns is fixed per dataset and should be treated as an ordered feature vector.

## Naming Convention

### EyeGazeDataset files

File format:

- `regions_MMDD_sampleIndex.csv`

Examples:

- `regions_1029_0.csv`
- `regions_1103_7.csv`

Interpretation:

- `MMDD` is the recording date (month/day), e.g. `1029` = Oct 29, `1103` = Nov 3
- `sampleIndex` is an integer segment/session index for that date

### EyesOpenCloseDataset files

File format:

- `openclose_MMDD_sampleIndex.csv`

Examples:

- `openclose_1104_0.csv`
- `openclose_1110_11.csv`

Interpretation:

- `MMDD` is the recording date (month/day)
- `sampleIndex` is an integer segment/session index for that date

## User Folders

`user1`, `user2`, `user3`, ... indicate participant-specific collections.

## Usage and Citation

If you use these data in publications, benchmark studies, or derivative datasets, please cite our IEEE Access paper.
