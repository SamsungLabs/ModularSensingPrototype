# EyeGazeDataset

Eye gaze region classification dataset organized by participant.

## Structure

- `user*/regions_MMDD_sampleIndex.csv`

### Example

- `user1/regions_1103_4.csv`

## Data schema

Each CSV contains:

- `Label`: target class for the sample row
- `Data_0 ... Data_N`: ordered numeric features

Each row is one sample.

## Filename decoding

- `regions`: dataset type (gaze region)
- `MMDD`: recording date
- `sampleIndex`: index of recording chunk/session for that date

## Citation

Please cite our IEEE Access paper if this dataset is used.
