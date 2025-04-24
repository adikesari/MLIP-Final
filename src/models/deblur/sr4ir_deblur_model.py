import torch
import torch.nn.functional as F
import os
import numpy as np
from ..base_model import BaseModel
from ...archs import build_network
from ...archs.common.bicubic_arch import BICUBIC
from ...utils.common import tensor2img

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

class SR4IRDeblurModel(BaseModel):
    """Super-Resolution model for Image Deblurring."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # define bicubic downsampling
        self.net_down = self.model_to_device(BICUBIC(scale=1/self.scale), is_trainable=False)
        
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
        
        # Create output directory for test results in experiments folder
        self.test_output_dir = os.path.join('experiments', opt['model_type'], opt['name'], 'test_results')
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create log file for PSNR results
        self.psnr_log_path = os.path.join('experiments', opt['model_type'], opt['name'], 'psnr_results.txt')
        with open(self.psnr_log_path, 'w') as f:
            f.write('Image Name,PSNR\n')

    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            self.net_deblur.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_deblur.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")

    def forward(self, x, target=None):
        # Downsample input using bicubic
        x_down = self.net_down(x)
        
        # Upsample input
        x_up = self.net_up(x_down)
        
        # Super-resolution
        x_sr = self.net_sr(x_up)
        
        # SR-to-HR comparison if target is provided
        if target is not None:
            # Create CQMix mask
            batch_size = x_sr.shape[0]
            mask = F.interpolate(
                (torch.randn(batch_size, 1, 8, 8)).bernoulli_(p=0.5),
                size=x_sr.shape[2:],
                mode='nearest'
            ).to(x_sr.device)
            
            # Apply CQMix
            x_cqmix = x_sr * mask + target * (1 - mask)
            
            # Deblurring on CQMix
            x_deblur = self.net_deblur(x_cqmix)
        else:
            # Regular deblurring without CQMix
            x_deblur = self.net_deblur(x_sr)
        
        return x_deblur, x_sr

    def optimize_parameters(self, current_iter):
        # Forward pass
        self.output, self.sr_output = self.forward(self.input, self.target)
        
        # Calculate loss
        self.loss_dict = self.calculate_loss()
        
        # Backward pass
        self.optimizer.zero_grad()
        self.loss_dict['total_loss'].backward()
        self.optimizer.step()
        
        return self.loss_dict

    def calculate_loss(self):
        loss_dict = {}
        
        # L1 loss for deblurred output
        loss_dict['l1_loss'] = self.criterion['l1'](self.output, self.target)
        
        # L1 loss for SR output
        loss_dict['sr_loss'] = self.criterion['l1'](self.sr_output, self.target)
        
        # Perceptual loss
        if 'perceptual' in self.criterion:
            loss_dict['perceptual_loss'] = self.criterion['perceptual'](self.output, self.target)
        
        # TDP loss
        if 'tdp' in self.criterion:
            loss_dict['tdp_loss'] = self.criterion['tdp'](self.output, self.target)
        
        # Total loss
        loss_dict['total_loss'] = sum(loss_dict.values())
        
        return loss_dict
        
    def test(self, data_loader, current_iter):
        """Test the model and save results."""
        self.set_mode('eval')
        total_psnr = 0
        num_images = 0
        
        for idx, data in enumerate(data_loader):
            self.input = data['blurry'].to(self.device)
            self.target = data['sharp'].to(self.device)
            
            with torch.no_grad():
                self.output, self.sr_output = self.forward(self.input)
            
            # Save results and calculate PSNR
            for i in range(self.input.size(0)):
                img_name = os.path.splitext(os.path.basename(data['blurry_path'][i]))[0]
                
                # Convert tensors to images
                output_img = tensor2img(self.output[i])
                target_img = tensor2img(self.target[i])
                
                # Calculate PSNR
                psnr = calculate_psnr(output_img, target_img)
                total_psnr += psnr
                num_images += 1
                
                # Log PSNR
                with open(self.psnr_log_path, 'a') as f:
                    f.write(f'{img_name},{psnr:.2f}\n')
                
                # Save images
                input_img = tensor2img(self.input[i])
                input_path = os.path.join(self.test_output_dir, f'{img_name}_input.png')
                input_img.save(input_path)
                
                sr_img = tensor2img(self.sr_output[i])
                sr_path = os.path.join(self.test_output_dir, f'{img_name}_sr.png')
                sr_img.save(sr_path)
                
                output_path = os.path.join(self.test_output_dir, f'{img_name}_deblur.png')
                output_img.save(output_path)
                
                target_path = os.path.join(self.test_output_dir, f'{img_name}_target.png')
                target_img.save(target_path)
        
        # Calculate and log average PSNR
        avg_psnr = total_psnr / num_images
        with open(self.psnr_log_path, 'a') as f:
            f.write(f'\nAverage PSNR: {avg_psnr:.2f}\n')
        
        print(f'Average PSNR: {avg_psnr:.2f} dB') 