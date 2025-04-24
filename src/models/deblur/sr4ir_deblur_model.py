import torch
from base_model import BaseModel
from ...archs import build_network

class SR4IRDeblurModel(BaseModel):
    """Super-Resolution model for Image Deblurring."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # define network up
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
        
        # define network sr
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=True)
        self.print_network(self.net_sr, tag='net_sr')
        
        # define network deblur
        self.net_deblur = build_network(opt['network_deblur'], self.text_logger, task='common', tag='net_deblur')
        self.load_network(self.net_deblur, name='network_deblur', tag='net_deblur')
        self.net_deblur = self.model_to_device(self.net_deblur, is_trainable=True)
        self.print_network(self.net_deblur, tag='net_deblur')

    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            self.net_deblur.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_deblur.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")

    def forward(self, x):
        # Upsample input
        x_up = self.net_up(x)
        
        # Super-resolution
        x_sr = self.net_sr(x_up)
        
        # Deblurring
        x_deblur = self.net_deblur(x_sr)
        
        return x_deblur

    def optimize_parameters(self, current_iter):
        # Forward pass
        self.output = self.forward(self.input)
        
        # Calculate loss
        self.loss_dict = self.calculate_loss()
        
        # Backward pass
        self.optimizer.zero_grad()
        self.loss_dict['total_loss'].backward()
        self.optimizer.step()
        
        return self.loss_dict

    def calculate_loss(self):
        loss_dict = {}
        
        # L1 loss
        loss_dict['l1_loss'] = self.criterion['l1'](self.output, self.target)
        
        # Perceptual loss
        if 'perceptual' in self.criterion:
            loss_dict['perceptual_loss'] = self.criterion['perceptual'](self.output, self.target)
        
        # TDP loss
        if 'tdp' in self.criterion:
            loss_dict['tdp_loss'] = self.criterion['tdp'](self.output, self.target)
        
        # Total loss
        loss_dict['total_loss'] = sum(loss_dict.values())
        
        return loss_dict 