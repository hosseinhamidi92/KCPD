# Dataset Access

The physiological dataset used in this paper is publicly available:

**Physiological Signals from Simulated Driving with Affective Events**

- **DOI**: [10.21227/x51t-jr44](https://dx.doi.org/10.21227/x51t-jr44)
- **Format**: Biopac `.txt` exports (2 kHz) + event timestamp spreadsheets
- **Subjects**: 15 participants
- **Modalities**: ECG (200 Hz), PPG (200 Hz), Resp (100 Hz), EDA (4 Hz), Skin Temp (4 Hz)
- **Duration**: ~30 min per session (Video → Survey → Practice → Survey → Experiment → Survey)

## Directory Structure

After downloading, place the data files as follows:

```
data/
├── P4S.txt
├── P5S.txt
├── ...
├── P32S.txt
└── timestamps.xlsx
```

## WESAD Cross-Dataset Validation

For the WESAD cross-dataset results (Section IV.D), download:

- **WESAD**: [https://archive.ics.uci.edu/dataset/465/wesad](https://archive.ics.uci.edu/dataset/465/wesad)

Place WESAD pickle files in `data/WESAD/S2/`, `data/WESAD/S3/`, etc.
