import torch
import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, scale=4, n_blocks=16, n_feats=64, res_scale=1.0):
        super(EDSR, self).__init__()
        self.scale = scale
        
        # define head module
        m_head = [nn.Conv2d(3, n_feats, kernel_size=3, padding=1)]
        
        # define body module
        m_body = []
        for _ in range(n_blocks):
            m_body.append(ResBlock(n_feats, res_scale))
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        
        # define tail module
        m_tail = [
            nn.Conv2d(n_feats, n_feats * (self.scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res *= self.res_scale
        return res + x

def build_network(**kwargs):
    return EDSR(**kwargs) 