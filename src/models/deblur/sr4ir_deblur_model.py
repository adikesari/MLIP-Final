import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np

from archs import build_network
from losses import build_loss
from torch.nn.functional import interpolate
from utils.common import save_on_master, quantize, calculate_psnr_batch, visualize_image, calculate_lpips_batch
from utils.det import MetricLogger, SmoothedValue

from .base_model import BaseModel


def make_model(opt):
    return SR4IRDeblurModel(opt)


class SR4IRDeblurModel(BaseModel):
    """Super-Resolution model for Image Deblurring."""

    def __init__(self, opt):
        super().__init__(opt)
        
        self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
        
        # define network sr
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, task='common', tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=True)
        self.print_network(self.net_sr, tag='net_sr')
        
        # define network deblur
        self.net_deblur = build_network(opt['network_deblur'], self.text_logger, task='common', tag='net_deblur')
        self.load_network(self.net_deblur, name='network_deblur', tag='net_deblur')
        self.net_deblur = self.model_to_device(self.net_deblur, is_trainable=True)
        self.print_network(self.net_deblur, tag='net_deblur')
        
        # Create output directory for test results
        self.test_output_dir = os.path.join('experiments', opt['model_type'], opt['name'], 'test_results')
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create log file for PSNR results
        self.psnr_log_path = os.path.join('experiments', opt['model_type'], opt['name'], 'psnr_results.txt')
        with open(self.psnr_log_path, 'w') as f:
            f.write('Image Name,PSNR\n')
        
        # Set random seed for CQMix
        torch.manual_seed(100)
        np.random.seed(100)
        
        # Initialize phase
        self.current_phase = 1

    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            self.net_deblur.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_deblur.eval()
        else:
            raise NotImplementedError(f"mode {mode} is not supported")

    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']

        # phase 1
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)
            
        if train_opt.get('tdp_opt'):
            # task driven loss
            self.cri_tdp = build_loss(train_opt['tdp_opt'], self.text_logger).to(self.device)
            
        # phase 2
        if train_opt.get('deblur_sr_opt'):
            self.cri_deblur_sr = build_loss(train_opt['deblur_sr_opt'], self.text_logger).to(self.device)
        
        if train_opt.get('deblur_hr_opt'):
            self.cri_deblur_hr = build_loss(train_opt['deblur_hr_opt'], self.text_logger).to(self.device)
            
        if train_opt.get('deblur_cqmix_opt'):
            self.cri_deblur_cqmix = build_loss(train_opt['deblur_cqmix_opt'], self.text_logger).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
        self.setup_schedulers(len(data_loader_train), name='deblur', optimizer=self.optimizer_deblur)
        
        # set up saving directories
        os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)
        
        # warmup
        self.warmup_epoch = train_opt.get('warmup_epoch', -1)
        self.text_logger.write("NOTICE: total epoch: {}, warmup epoch: {}".format(train_opt['epoch'], self.warmup_epoch))

    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer sr
        optim_type = train_opt['optim_sr'].pop('type')
        self.optimizer_sr = self.get_optimizer(optim_type, self.net_sr.parameters(), **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)
        
        # optimizer deblur
        optim_type = train_opt['optim_deblur'].pop('type')
        net_deblur_parameters = [p for p in self.net_deblur.parameters() if p.requires_grad]
        self.optimizer_deblur = self.get_optimizer(optim_type, net_deblur_parameters, **train_opt['optim_deblur'])
        self.optimizers.append(self.optimizer_deblur)

    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        self.set_mode(mode='train')
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("lr_deblur", SmoothedValue(window_size=1, fmt="{value}"))
        
        if self.dist:
            train_sampler.set_epoch(epoch)
            
        if epoch < self.warmup_epoch + 1:
            self.text_logger.write("NOTICE: Doing warm-up")
            
        # NOTE: without warmup, training explodes!!
        lr_scheduler_s = None
        lr_scheduler_d = None
        if epoch == 1:
            warmup_factor = 1.0 / len(data_loader_train)
            warmup_iters = len(data_loader_train)
            lr_scheduler_s = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_sr, start_factor=warmup_factor, total_iters=warmup_iters)
            lr_scheduler_d = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_deblur, start_factor=warmup_factor, total_iters=warmup_iters)

        header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
        for iter, (img_hr_list, target_list) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
            blurry_list = [t['blurry'] for t in target_list]
            sharp_list = [t['sharp'] for t in target_list]
            
            blurry_list = list(img.to(self.device) for img in blurry_list)
            sharp_list = list(img.to(self.device) for img in sharp_list)
            current_iter = iter + len(data_loader_train)*(epoch-1)

            # make on-the-fly LR image from blurry image
            blurry_batch = self.list_to_batch(blurry_list)
            img_lr_batch = quantize(interpolate(blurry_batch, scale_factor=(1/self.scale), mode='bicubic'))
            
            # phase 1
            img_sr_batch = self.net_sr(img_lr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=blurry_list)
            for p in self.net_deblur.parameters(): p.requires_grad = False
            self.optimizer_sr.zero_grad()
            l_total_sr = 0

            # Pixel loss
            if hasattr(self, 'cri_pix'):
                img_sr_batch_resized = F.interpolate(img_sr_batch, size=blurry_batch.shape[2:], mode='bilinear', align_corners=False)
                l_pix = self.cri_pix(img_sr_batch_resized, blurry_batch)
                metric_logger.meters["l_pix"].update(l_pix.item()) 
                self.tb_logger.add_scalar('losses/l_pix', l_pix.item(), current_iter)
                l_total_sr += l_pix
                del img_sr_batch_resized

            # TDP loss
            if epoch > self.warmup_epoch:
                if hasattr(self, 'cri_tdp'):
                    l_tdp = self.cri_tdp(img_sr_batch, blurry_batch)
                    metric_logger.meters["l_tdp"].update(l_tdp.item())
                    self.tb_logger.add_scalar('losses/l_tdp', l_tdp.item(), current_iter)
                    l_total_sr += l_tdp

            l_total_sr.backward()
            self.optimizer_sr.step()
            
            # Clear memory after phase 1
            torch.cuda.empty_cache()
            
            # phase 2
            img_sr_batch = self.net_sr(img_lr_batch).detach()
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=blurry_list)
            for p in self.net_deblur.parameters(): p.requires_grad = True
            self.optimizer_deblur.zero_grad()
            l_total_deblur = 0

            # Prepare inputs for deblurring
            img_sr_batch = self.list_to_batch(img_sr_list)
            blurry_batch = self.list_to_batch(blurry_list)
            blurry_batch = F.interpolate(blurry_batch, size=img_sr_batch.shape[2:], mode='bilinear', align_corners=False)
            sharp_batch = self.list_to_batch(sharp_list)
            
            # Create CQMix image
            batch_size = len(blurry_list)
            mask = interpolate((torch.randn(batch_size,1,8,8)).bernoulli_(p=0.5), size=(img_sr_batch.shape[2:]), mode='nearest').to(self.device)
            img_cqmix_batch = img_sr_batch*mask + blurry_batch*(1-mask)
            # Calculate SR loss
            if hasattr(self, 'cri_deblur_sr'):
                output_sr = self.net_deblur(img_sr_batch)
                output_sr = output_sr + img_sr_batch
                l_deblur_sr = self.cri_deblur_sr(output_sr, img_sr_batch)
                metric_logger.meters["l_deblur_sr"].update(l_deblur_sr.item())
                self.tb_logger.add_scalar('losses/l_deblur_sr', l_deblur_sr.item(), current_iter)
                l_total_deblur += l_deblur_sr
                del output_sr

            # Calculate HR loss
            if hasattr(self, 'cri_deblur_hr'):
                output_hr = self.net_deblur(blurry_batch)
                output_hr = output_hr + blurry_batch
                l_deblur_hr = self.cri_deblur_hr(output_hr, blurry_batch)
                metric_logger.meters["l_deblur_hr"].update(l_deblur_hr.item())
                self.tb_logger.add_scalar('losses/l_deblur_hr', l_deblur_hr.item(), current_iter)
                l_total_deblur += l_deblur_hr
                del output_hr

            # Calculate CQMix loss
            if hasattr(self, 'cri_deblur_cqmix'):
                output_cqmix = self.net_deblur(img_cqmix_batch)
                output_cqmix = output_cqmix + img_cqmix_batch
                l_deblur_cqmix = self.cri_deblur_cqmix(output_cqmix, img_cqmix_batch)
                metric_logger.meters["l_deblur_cqmix"].update(l_deblur_cqmix.item())
                self.tb_logger.add_scalar('losses/l_deblur_cqmix', l_deblur_cqmix.item(), current_iter)
                l_total_deblur += l_deblur_cqmix
                del output_cqmix

            l_total_deblur.backward()
            self.optimizer_deblur.step()
            # logging training state
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), sharp_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            metric_logger.update(lr_sr=round(self.optimizer_sr.param_groups[0]["lr"], 8))
            metric_logger.update(lr_deblur=round(self.optimizer_deblur.param_groups[0]["lr"], 8))
            
            if epoch == 1:
                lr_scheduler_s.step()
                lr_scheduler_d.step()
            else:
                self.update_learning_rate()
                
            # Clear all memory (helps our GPU)
            del img_sr_batch, blurry_batch, sharp_batch, img_lr_batch, img_sr_list, img_cqmix_batch
            torch.cuda.empty_cache()
        return

    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0):
        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return
        
        self.set_mode(mode='eval')
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"
        
        num_processed_samples = 0
        for img_hr_list, target_list in metric_logger.log_every(data_loader_test, 1000, self.text_logger, header):
            img_hr_list = list(img_hr.to(self.device) for img_hr in img_hr_list)
            target_list = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_list]

            # make on-the-fly LR image
            img_hr_batch = self.list_to_batch(img_hr_list)
            img_lr_batch = quantize(interpolate(img_hr_batch, scale_factor=(1/self.scale), mode='bicubic'))
            
            # perform SR
            img_sr_batch = self.net_sr(img_lr_batch)
            img_sr_list = self.batch_to_list(img_sr_batch, img_list=img_hr_list)
            
            # deblurring
            if torch.cuda.is_available(): torch.cuda.synchronize()
            # Batch the SR images for deblurring
            img_sr_batch = self.list_to_batch(img_sr_list)
            img_hr_batch = self.list_to_batch([t['sharp'] for t in target_list])
            img_hr_batch = F.interpolate(img_hr_batch, size=img_sr_batch.shape[2:], mode='bilinear', align_corners=False)
            # Create CQMix image
            batch_size = len(img_hr_list)
            mask = interpolate((torch.randn(batch_size,1,8,8)).bernoulli_(p=0.5), size=(img_sr_batch.shape[2:]), mode='nearest').to(self.device)
            img_cqmix_batch = img_sr_batch*mask + img_hr_batch*(1-mask)
            # Concatenate all three images (SR, HR, CQMix)
            img_input = torch.cat([img_sr_batch, img_hr_batch, img_cqmix_batch], dim=1)
            output = self.net_deblur(img_input)
            output = F.interpolate(output, size=img_hr_batch.shape[2:], mode='bilinear', align_corners=False)
            outputs_sr = self.batch_to_list(output, img_list=img_hr_list)

            # visualizing
            if self.opt.get('test_only', False):
                for i, (img_sr, output_sr) in enumerate(zip(img_sr_list, outputs_sr)):
                    # Get the original filename from the dataset
                    original_filename = os.path.splitext(os.path.basename(data_loader_test.dataset.image_list[num_processed_samples + i][1]))[0]
                    self.visualize(img_sr, output_sr, original_filename)

            # evaluation on validation batch
            batch_size = len(img_sr_list)
            psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr_batch), img_hr_batch)
            metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
            if self.opt['test'].get('calculate_lpips', False):
                lpips, valid_batch_size = calculate_lpips_batch(quantize(img_sr_batch), img_hr_batch, self.net_lpips)
                metric_logger.meters["lpips"].update(lpips.item(), n=valid_batch_size)
            num_processed_samples += batch_size
    
        metric_logger.synchronize_between_processes()
        
        # logging training state
        metric_summary = f"{header}"
        metric_summary = self.add_metric(metric_summary, 'PSNR', metric_logger.psnr.global_avg, epoch)
        if self.opt['test'].get('calculate_lpips', False):
            metric_summary = self.add_metric(metric_summary, 'LPIPS', metric_logger.lpips.global_avg, epoch)
        self.text_logger.write(metric_summary)
        return

    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_sr": self.get_bare_model(self.net_sr).state_dict(),
                      "net_deblur": self.get_bare_model(self.net_deblur).state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_{:03d}.pth".format(epoch)))
            save_on_master(self.get_bare_model(self.net_deblur).state_dict(), osp.join(self.exp_dir, 'models', "net_deblur_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_latest.pth"))
        save_on_master(self.get_bare_model(self.net_deblur).state_dict(), osp.join(self.exp_dir, 'models', "net_deblur_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return 