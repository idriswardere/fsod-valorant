# This class contains the trainer proposed by our project

import torch
from ..tools.train_net import Trainer

class NewTrainer(Trainer):
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        lr = 0.001
        layer_decay_factor = 0.5
        decay_num = 10
        parameters = []
        layer_names = []
        for idx, (name, param) in enumerate(model.named_parameters()):
            layer_names.append(name)
        prev_group_name = layer_names[-1].split('.')[-2]
        for name in layer_names[::-1]:
            # # parameter group name
            cur_group_name = name.split('.')[-2]
            #print(cur_group_name)
            # # update learning rate
            if cur_group_name != prev_group_name:
                lr *= layer_decay_factor
                decay_num -= 1
            prev_group_name = cur_group_name
            # # display info
            # # append layer parameters # order problem !!
            parameters.append({'params': [p for n, p in model.named_parameters() if n == name ],
                            'lr':     lr,
                            'dampening': 0,
                            'momentum': 0.9,
                            'weight_decay': 0.0001,
                            'foreach': True})
            if decay_num == 0: break
        optimizer = torch.optim.SGD(parameters)
        del parameters
        return optimizer
    
