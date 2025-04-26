import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import VOCDetection
from utils.det import DetectionPresetTrain, DetectionPresetEval, get_coco, create_aspect_ratio_groups, GroupedBatchSampler, collate_fn

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

def load_deblur_data(opt):
    """Load training and validation datasets for deblurring task."""
    use_trainset = opt.get('train', False)
    data_format = opt['data']['format']
    
    # transform
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=opt['datasets']['train']['mean'], std=opt['datasets']['train']['std'])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=opt['datasets']['val']['mean'], std=opt['datasets']['val']['std'])
    ])
    
    # datasets
    dataset_train = None
    if use_trainset:
        train_opt = opt['datasets']['train'].copy()
        train_opt['transform'] = transform_train
        dataset_train = DeblurDataset(train_opt)
            
    val_opt = opt['datasets']['val'].copy()
    val_opt['transform'] = transform_test
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
            dataset_train, batch_sampler=train_batch_sampler, num_workers=opt['num_threads'], pin_memory=True)

    data_loader_test = None
    if opt.get('test', False):
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=opt['num_threads'], pin_memory=True)

    return data_loader_train, data_loader_test, train_sampler, test_sampler 