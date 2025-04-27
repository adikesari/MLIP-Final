from .cls import load_cls_data
from .det import load_det_data
from .seg import load_seg_data
from .deblur_dataset import load_deblur_data
import torch
import torch.utils.data as data
from .deblur_dataset import DeblurDataset

def load_data(opt):
    task = opt.get('task', 'cls').lower()
    print('Data path: {}'.format(opt['data']['path']))
    return eval(f"load_{task}_data")(opt)
