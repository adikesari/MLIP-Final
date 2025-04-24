import torch
import torch.nn as nn

class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.scale = opt.get('scale', 1)
        self.text_logger = opt.get('text_logger', None)
        self.criterion = {}
        self.optimizer = None
        self.input = None
        self.target = None
        self.output = None

    def model_to_device(self, model, is_trainable=True):
        model = model.to(self.opt['device'])
        if not is_trainable:
            model.eval()
        return model

    def load_network(self, network, name, tag):
        if self.opt.get('pretrained_models', None) is not None:
            if name in self.opt['pretrained_models']:
                network.load_state_dict(torch.load(self.opt['pretrained_models'][name]))

    def print_network(self, network, tag):
        if self.text_logger is not None:
            self.text_logger.info(f'Network {tag}:\n{network}') 