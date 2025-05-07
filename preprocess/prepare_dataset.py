import os
import urllib.request
import tarfile
import shutil
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

def copy_original_images(input_dir, output_dir, image_list):
    print(f"Copying original images to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy each image
    for img_name in image_list:
        img_name = img_name.strip()  # Remove newline
        src_path = os.path.join(input_dir, f"{img_name}.jpg")
        dst_path = os.path.join(output_dir, f"{img_name}.jpg")
        shutil.copy2(src_path, dst_path)

def main():
    # Set up directories
    base_dir = Path("datasets")
    voc_dir = base_dir / "VOC2012"
    deblur_dir = base_dir / "deblur"
    
    voc_path = download_voc_dataset(str(voc_dir))
    train_dir = deblur_dir / "train"
    val_dir = deblur_dir / "val"

    train_sharp_dir = train_dir / "sharp_images"
    val_sharp_dir = val_dir / "sharp_images"
    
    train_list_path = os.path.join(voc_path, "ImageSets", "Main", "train.txt")
    val_list_path = os.path.join(voc_path, "ImageSets", "Main", "val.txt")
    
    with open(train_list_path) as f:
        train_images = f.readlines()
    with open(val_list_path) as f:
        val_images = f.readlines()
    
    copy_original_images(
        os.path.join(voc_path, "JPEGImages"),
        str(train_sharp_dir),
        train_images
    )
    copy_original_images(
        os.path.join(voc_path, "JPEGImages"),
        str(val_sharp_dir),
        val_images
    )
    
    print("Processing training set...")
    process_voc_dataset(
        input_dir=os.path.join(voc_path, "JPEGImages"),
        output_dir=str(train_dir / "blurry_images"),
        image_list=train_images,
        blur_lengths=[5, 9, 11],
        angles=[0, 45, 90]
    )
    
    print("Processing validation set...")
    process_voc_dataset(
        input_dir=os.path.join(voc_path, "JPEGImages"),
        output_dir=str(val_dir / "blurry_images"),
        image_list=val_images,
        blur_lengths=[7, 9],  # Fewer variations for validation
        angles=[0, 90]
    )

if __name__ == "__main__":
    main() 