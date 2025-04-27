import torch
import torch.nn as nn
import torch.nn.functional as F

class NAFNet(nn.Module):
    def __init__(self, img_channel=9, width=32, middle_blk_num=12, 
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super(NAFNet, self).__init__()
        self.img_channel = img_channel
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums
        
        # define head module
        self.head = nn.Conv2d(self.img_channel, self.width, kernel_size=3, padding=1)
        
        # define encoder
        self.encoders = nn.ModuleList()
        for i in range(len(self.enc_blk_nums)):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(self.width * (2 ** i)) for _ in range(self.enc_blk_nums[i])]
                )
            )
        
        # define middle
        self.middle = nn.Sequential(
            *[NAFBlock(self.width * (2 ** len(self.enc_blk_nums))) for _ in range(self.middle_blk_num)]
        )
        
        # define decoder
        self.decoders = nn.ModuleList()
        for i in range(len(self.dec_blk_nums)):
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(self.width * (2 ** (len(self.enc_blk_nums) - i - 1))) for _ in range(self.dec_blk_nums[i])]
                )
            )
        
        # define tail module
        self.tail = nn.Conv2d(self.width, 3, kernel_size=3, padding=1)
        
    def forward(self, x, target=None):
        # head
        x = self.head(x)
        
        # encoder
        enc_feats = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            enc_feats.append(x)
            if i < len(self.encoders) - 1:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # middle
        x = self.middle(x)
        
        # decoder
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = x + enc_feats[-(i+1)]
            x = decoder(x)
        
        # tail
        x = self.tail(x)
        
        if target is not None:
            # Calculate loss if target is provided
            loss = F.l1_loss(x, target['sharp'])
            return x, {'total_loss': loss}
        
        return x

class NAFBlock(nn.Module):
    def __init__(self, dim):
        super(NAFBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        
        return out + identity

def build_network(**kwargs):
    return NAFNet(**kwargs) 