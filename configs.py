from collections.abc import Mapping
from copy import deepcopy

class PersistentDict(Mapping):
    #https://en.wikipedia.org/wiki/Persistent_data_structure
    def __init__(self, **kwargs):
        self.data = {}

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return self.data.__len__()

    def __setitem__(self, key, value):
        self.data = deepcopy(self.data)
        return self.data.__setitem__(key, value)


def vary_config(baseconfig, mode='combinatorial', **kwargs):
    """With baseconfig as the default, return a list of configs with entries given by kwargs.keys()
    varied according to the corresponding lists in kwargs.values(). Baseconfig is not included"""
    pass


###################
# Default configs #
###################
# Training
train_config = {'name': 'FPT', #FPT, SGD
                'batch_size': 50,
                'lr': 0.1, #for FPT only
                'print_every': 10,
                'loss_mode': 'full', # full, class
                }


# Networks
net_config = {'name': 'ModernHopfield',
              'input_size': 794,
              'hidden_size': 50,
              'beta': 1,
              'tau': 1,
              'input_mode': 'init', #init, cont, init+cont, clamp
              'dt': 0.05,
              'num_steps': 1000,
              'fp_mode': 'iter', #iter, del2
              'fp_thres':1e-9,
              }


# Datasets
data_config = {'name': 'MNIST',
               'test': True
               }



###################
# Premade configs #
###################
def get_config(name='default'):
    if name == 'default':
        config = {'net': net_config,
                  'data': data_config,
                  'train': train_config}
    else:
        raise ValueError(f"Invalid config name: '{name}'")

    return deepcopy(config)
