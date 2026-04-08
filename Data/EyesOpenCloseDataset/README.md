# EyesOpenCloseDataset

Eye open/close state classification dataset organized by participant.

## Structure

- `user*/openclose_MMDD_sampleIndex.csv`

### Example

- `user1/openclose_1106_3.csv`

## Data schema

Each CSV contains:

- `Label`: target class for the sample row
- `Data_0 ... Data_N`: ordered numeric features

Each row is one sample.

## Filename decoding

- `openclose`: dataset type (eye open/close)
- `MMDD`: recording date
- `sampleIndex`: index of recording chunk/session for that date

## Citation

Please cite our IEEE Access paper if this dataset is used.
