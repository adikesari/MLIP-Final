from .cls import load_cls_data
from .det import load_det_data
from .seg import load_seg_data
import torch
import torch.utils.data as data
from .deblur_dataset import DeblurDataset

def load_data(opt):
    """Load training and validation datasets."""
    # Training dataset
    train_dataset = DeblurDataset(opt['datasets']['train'])
    train_sampler = None
    if opt.get('distributed', False):
        train_sampler = data.distributed.DistributedSampler(train_dataset)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=opt['train']['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=opt['datasets']['train']['num_workers'],
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True
    )
    
    # Validation dataset
    val_dataset = DeblurDataset(opt['datasets']['val'])
    val_sampler = None
    if opt.get('distributed', False):
        val_sampler = data.distributed.DistributedSampler(val_dataset)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=opt['datasets']['val']['batch_size'],
        shuffle=False,
        num_workers=opt['datasets']['val']['num_workers'],
        sampler=val_sampler,
        drop_last=False,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler, val_sampler
