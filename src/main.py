import datetime
import os.path as osp
import time
import torch
import torch.utils.data
import utils.common
import argparse

from models import make_model
from data import load_data


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--test_only', action='store_true', help='Run testing only')
    args = parser.parse_args()
    
    # load opt and args from yaml
    opt, _ = utils.common.parse_options(args.opt)
    
    # Set device
    if opt.get('device') == 'cuda':
        if not torch.cuda.is_available():
            print('CUDA is not available, using CPU instead')
            opt['device'] = 'cpu'
        else:
            torch.cuda.set_device(opt['gpu_ids'][0])
    
    # Initialize distributed training if needed
    if opt.get('distributed', False):
        opt = utils.common.init_distributed_mode(opt)
    
    # deterministic option for reproduction
    if opt.get('deterministic', False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        
    # make model
    model = make_model(opt)
    utils.common.copy_opt_file(args.opt, osp.join('experiments', opt['model_type'], opt['name']))
    
    # prepare data loader
    data_loader_train, data_loader_test, train_sampler, test_sampler = load_data(opt)
    
    if args.test_only:
        # Testing mode
        print('Testing mode...')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        if opt.get('calculate_cost'):
            model.calculate_cost()
        else:
            model.test(data_loader_test, 0)
    else:
        # Training mode
        if opt.get('train', False):
            model.init_training_settings(data_loader_train)
            if opt.get('resume', False):
                resume_epoch = model.resume_training(opt['resume'])
                start_epoch, end_epoch = resume_epoch+1, opt['train']['epoch']
            else:
                start_epoch, end_epoch = 1, opt['train']['epoch']
            
            model.text_logger.write("Start training")
            start_time = time.time()
            
            for epoch in range(start_epoch, end_epoch+1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                
                # Training
                model.train_one_epoch(data_loader_train, train_sampler, epoch)
                
                # Evaluation
                if epoch % opt['train']['eval_freq'] == 0:
                    model.evaluate(data_loader_test, epoch)
                
                # Save checkpoint
                if epoch % opt['train']['save_freq'] == 0:
                    model.save(epoch)
                
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            model.text_logger.write(f"Training time {total_time_str}")
        else:
            print('No training configuration found in the YAML file.')


if __name__ == "__main__":
    main()
