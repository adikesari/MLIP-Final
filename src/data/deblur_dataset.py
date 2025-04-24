import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class DeblurDataset(data.Dataset):
    """Dataset for deblurring task."""
    
    def __init__(self, opt):
        super(DeblurDataset, self).__init__()
        self.opt = opt
        
        # Set paths
        self.data_root = opt['data_root']
        self.blurry_path = os.path.join(self.data_root, 'blurry_images')
        self.sharp_path = os.path.join(self.data_root, 'sharp_images')
        
        # Get image list
        self.image_list = self._get_image_list()
        
        # Set transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=opt['mean'], std=opt['std'])
        ])
        
        # Set resolutions
        self.l_resolution = opt['l_resolution']
        self.r_resolution = opt['r_resolution']
        
    def _get_image_list(self):
        """Get list of image pairs."""
        image_list = []
        for img_name in os.listdir(self.blurry_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                blurry_path = os.path.join(self.blurry_path, img_name)
                sharp_path = os.path.join(self.sharp_path, img_name)
                if os.path.exists(sharp_path):
                    image_list.append((blurry_path, sharp_path))
        return image_list
    
    def __getitem__(self, index):
        """Get a pair of blurry and sharp images."""
        blurry_path, sharp_path = self.image_list[index]
        
        # Load images
        blurry_img = Image.open(blurry_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        
        # Resize images
        blurry_img = blurry_img.resize((self.l_resolution, self.l_resolution), Image.BICUBIC)
        sharp_img = sharp_img.resize((self.r_resolution, self.r_resolution), Image.BICUBIC)
        
        # Apply transforms
        blurry_img = self.transform(blurry_img)
        sharp_img = self.transform(sharp_img)
        
        return {
            'blurry': blurry_img,
            'sharp': sharp_img,
            'blurry_path': blurry_path,
            'sharp_path': sharp_path
        }
    
    def __len__(self):
        return len(self.image_list) 