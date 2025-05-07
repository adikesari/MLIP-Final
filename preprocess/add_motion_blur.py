import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import random

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

def process_voc_dataset(input_dir, output_dir, image_list, blur_lengths=[5, 7, 9, 11], angles=[0, 45, 90, 135]):
    """Process VOC dataset images and add motion blur."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(image_list)} images")
    
    for img_name in tqdm(image_list):
        img_name = img_name.strip()  # Remove newline
        img_path = os.path.join(input_dir, f"{img_name}.jpg")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Randomly select blur parameters
        length = random.choice(blur_lengths)
        angle = random.choice(angles)
        
        # Apply motion blur
        kernel = create_motion_blur_kernel(length, angle)
        blurred_img = add_motion_blur(img, kernel)
        output_path = os.path.join(output_dir, f"{img_name}.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save blurred image
        cv2.imwrite(output_path, blurred_img)

def main():
    parser = argparse.ArgumentParser(description='Add motion blur to VOC dataset images')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing VOC dataset images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for blurred images')
    parser.add_argument('--image_list', type=str, required=True, help='Path to file containing list of images to process')
    parser.add_argument('--blur_lengths', type=int, nargs='+', default=[5, 7, 9, 11], help='List of blur kernel lengths')
    parser.add_argument('--angles', type=int, nargs='+', default=[0, 45, 90, 135], help='List of blur angles')
    
    args = parser.parse_args()
    
    with open(args.image_list) as f:
        image_list = f.readlines()
    
    process_voc_dataset(args.input_dir, args.output_dir, image_list, args.blur_lengths, args.angles)

if __name__ == '__main__':
    main() 