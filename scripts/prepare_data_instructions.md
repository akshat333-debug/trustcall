# ASVspoof 2019 Dataset Setup Instructions

This project uses the ASVspoof 2019 Logical Access (LA) dataset. Due to licensing and size, you must download it manually.

## 1. Download the Dataset
Go to the official ASVspoof 2019 website or the datashare link:
- **URL**: [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)
- **File**: `LA.zip` (Logical Access) ~5GB.

(Optional) For Physical Access, download `PA.zip`.

## 2. Extract and Organize
Extract the contents into the `data/asvspoof2019` directory. The structure should look like this:

```
data/
└── asvspoof2019/
    └── LA/
        ├── ASVspoof2019_LA_cm_protocols/
        │   ├── ASVspoof2019.LA.cm.train.trn.txt
        │   ├── ASVspoof2019.LA.cm.dev.trl.txt
        │   └── ASVspoof2019.LA.cm.eval.trl.txt
        ├── ASVspoof2019_LA_train/
        │   └── flac/ (contains .flac files)
        ├── ASVspoof2019_LA_dev/
        │   └── flac/
        └── ASVspoof2019_LA_eval/
            └── flac/
```

## 3. Verify
Run the verification script to ensure everything is in place:
```bash
python scripts/verify_dataset.py
```
This script will check file counts and protocol files.
