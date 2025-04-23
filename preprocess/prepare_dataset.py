import os
import urllib.request
import tarfile
from pathlib import Path
from add_motion_blur import process_voc_dataset

def download_voc_dataset(output_dir):
    """Download and extract VOC2012 dataset."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download VOC2012 dataset
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    tar_path = os.path.join(output_dir, "VOCtrainval_11-May-2012.tar")
    
    if not os.path.exists(tar_path):
        print("Downloading VOC2012 dataset...")
        urllib.request.urlretrieve(url, tar_path)
    
    # Extract dataset
    if not os.path.exists(os.path.join(output_dir, "VOCdevkit")):
        print("Extracting VOC2012 dataset...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=output_dir)
    
    return os.path.join(output_dir, "VOCdevkit", "VOC2012")

def main():
    # Set up directories
    base_dir = Path("datasets")
    voc_dir = base_dir / "VOC2012"
    deblur_dir = base_dir / "deblur"
    
    # Download and extract VOC dataset
    voc_path = download_voc_dataset(str(voc_dir))
    
    # Create train and val directories
    train_dir = deblur_dir / "train"
    val_dir = deblur_dir / "val"
    
    # Process training set
    print("Processing training set...")
    process_voc_dataset(
        input_dir=os.path.join(voc_path, "JPEGImages"),
        output_dir=str(train_dir),
        blur_lengths=[5, 7, 9, 11],
        angles=[0, 45, 90, 135]
    )
    
    # Process validation set
    print("Processing validation set...")
    process_voc_dataset(
        input_dir=os.path.join(voc_path, "JPEGImages"),
        output_dir=str(val_dir),
        blur_lengths=[7, 9],  # Fewer variations for validation
        angles=[0, 90]
    )

if __name__ == "__main__":
    main() 