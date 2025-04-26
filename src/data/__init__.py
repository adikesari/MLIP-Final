from .cls import load_cls_data
from .det import load_det_data
from .seg import load_seg_data
from .deblur_dataset import load_deblur_data
import torch
import torch.utils.data as data
from .deblur_dataset import DeblurDataset

def load_data(opt):
    """Load training and validation datasets."""
    if opt['model_type'] == 'sr4ir_det':
        return load_det_data(opt)
    elif opt['model_type'] == 'sr4ir_deblur':
        return load_deblur_data(opt)
    else:
        raise NotImplementedError(f"Model type {opt['model_type']} is not supported")
