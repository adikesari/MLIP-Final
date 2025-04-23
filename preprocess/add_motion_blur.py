import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

def create_motion_blur_kernel(length, angle):
    """Create a motion blur kernel with given length and angle."""
    kernel = np.zeros((length, length))
    kernel[length//2, :] = 1
    kernel = cv2.warpAffine(kernel, 
                           cv2.getRotationMatrix2D((length//2, length//2), angle, 1.0),
                           (length, length))
    kernel = kernel / kernel.sum()
    return kernel

def add_motion_blur(image, kernel):
    """Apply motion blur to an image using the given kernel."""
    return cv2.filter2D(image, -1, kernel)

def process_voc_dataset(input_dir, output_dir, blur_lengths=[5, 7, 9, 11], angles=[0, 45, 90, 135]):
    """Process VOC dataset images and add motion blur."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(input_dir).rglob(f'*{ext}')))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Create blurred versions
        for length in blur_lengths:
            for angle in angles:
                # Create blur kernel
                kernel = create_motion_blur_kernel(length, angle)
                
                # Apply blur
                blurred_img = add_motion_blur(img, kernel)
                
                # Create output filename
                rel_path = img_path.relative_to(input_dir)
                output_path = Path(output_dir) / rel_path.parent / f"{rel_path.stem}_blur_{length}_{angle}{rel_path.suffix}"
                
                # Create output directory if it doesn't exist
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Save blurred image
                cv2.imwrite(str(output_path), blurred_img)

def main():
    parser = argparse.ArgumentParser(description='Add motion blur to VOC dataset images')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing VOC dataset images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for blurred images')
    parser.add_argument('--blur_lengths', type=int, nargs='+', default=[5, 7, 9, 11], help='List of blur kernel lengths')
    parser.add_argument('--angles', type=int, nargs='+', default=[0, 45, 90, 135], help='List of blur angles')
    
    args = parser.parse_args()
    
    process_voc_dataset(args.input_dir, args.output_dir, args.blur_lengths, args.angles)

if __name__ == '__main__':
    main() 