import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from utils.det import create_aspect_ratio_groups, GroupedBatchSampler

class DeblurDataset(data.Dataset):
    """Dataset for deblurring task."""
    
    def __init__(self, opt):
        super(DeblurDataset, self).__init__()
        self.opt = opt
        
        self.data_root = opt['data_root']
        self.blurry_path = os.path.join(self.data_root, 'blurry_images')
        self.sharp_path = os.path.join(self.data_root, 'sharp_images')

        self.image_list = self._get_image_list()
        print(f"Found {len(self.image_list)} image pairs in {self.data_root}")
        
        # Set transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=opt['data']['mean'], std=opt['data']['std'])
        ])
        
        self.ids = [os.path.splitext(os.path.basename(path))[0] for path, _ in self.image_list]
        
        self.max_size = 360  # Maximum dimension size
        
    def _get_image_list(self):
        """Get list of image pairs."""
        image_list = []
        if not os.path.exists(self.blurry_path):
            raise FileNotFoundError(f"Blurry images directory not found: {self.blurry_path}")
        if not os.path.exists(self.sharp_path):
            raise FileNotFoundError(f"Sharp images directory not found: {self.sharp_path}")
            
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
        
        blurry_img = Image.open(blurry_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')
        
        def resize_image(img):
            w, h = img.size
            if max(w, h) > self.max_size:
                if w > h:
                    new_w = self.max_size
                    new_h = int(h * (self.max_size / w))
                else:
                    new_h = self.max_size
                    new_w = int(w * (self.max_size / h))
                img = img.resize((new_w, new_h), Image.BICUBIC)
            return img
        
        blurry_img = resize_image(blurry_img)
        sharp_img = resize_image(sharp_img)
        
        # Transform images to tensors
        blurry_img = self.transform(blurry_img)
        sharp_img = self.transform(sharp_img)
        
        # Create target dictionary with only tensor values
        target = {
            'sharp': sharp_img
        }
        
        return blurry_img, target
    
    def __len__(self):
        return len(self.image_list)

def load_deblur_data(opt):
    """Load training and validation datasets for deblurring task."""
    use_trainset = opt.get('train', False)
    data_format = opt['data']['format']
    
    # transform
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=opt['data']['mean'], std=opt['data']['std'])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=opt['data']['mean'], std=opt['data']['std'])
    ])
    
    # datasets
    dataset_train = None
    if use_trainset:
        train_opt = opt['data']['train'].copy()
        train_opt['transform'] = transform_train
        train_opt['data'] = opt['data']  # Add data section for mean/std
        dataset_train = DeblurDataset(train_opt)
            
    val_opt = opt['data']['val'].copy()
    val_opt['transform'] = transform_test
    val_opt['data'] = opt['data']  # Add data section for mean/std
    dataset_test = DeblurDataset(val_opt)

    # distributed training
    train_sampler = None
    if opt.get('distributed', False):
        if use_trainset:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if use_trainset:
            train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    if use_trainset:
        # aspect ratio batch sampler
        if opt['data'].get('aspect_ratio_group_factor', 3) >= 0:
            group_ids = create_aspect_ratio_groups(dataset_train, k=opt['data']['aspect_ratio_group_factor'])
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, opt['train']['batch_size'])
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, opt['train']['batch_size'], drop_last=True)  
    
    # data loader    
    data_loader_train = None
    if use_trainset:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            batch_sampler=train_batch_sampler, 
            num_workers=opt['num_threads'], 
            pin_memory=True,
            collate_fn=lambda x: (
                [item[0] for item in x],  # List of blurry images
                [item[1] for item in x]   # List of target dictionaries
            )
        )

    data_loader_test = None
    if opt.get('test', False):
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=1, 
            sampler=test_sampler, 
            num_workers=opt['num_threads'], 
            pin_memory=True,
            collate_fn=lambda x: (
                [item[0] for item in x],  # List of blurry images
                [item[1] for item in x]   # List of target dictionaries
            )
        )

    return data_loader_train, data_loader_test, train_sampler, test_sampler 