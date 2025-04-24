# Dataset Preprocessing

This directory contains scripts for preprocessing the VOC dataset to create a deblurring dataset.

## Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the preprocessing script:
```bash
python prepare_dataset.py
```

This will:
- Download the VOC2012 dataset
- Extract it to the `datasets/VOC2012` directory
- Create blurred versions of the images
- Save them to `datasets/deblur/train` and `datasets/deblur/val`

## Blur Parameters

Training set:
- Blur lengths: [5, 7, 9, 11] pixels
- Angles: [0°, 45°, 90°, 135°]

Validation set:
- Blur lengths: [7, 9] pixels
- Angles: [0°, 90°]

## Customization

You can modify the blur parameters by editing the `prepare_dataset.py` script:
- Change blur lengths
- Modify angles
- Add different blur types
- Adjust the train/val split 