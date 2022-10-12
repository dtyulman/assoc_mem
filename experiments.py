from copy import deepcopy
import networks
import configs as cfg

####################
# Pre-made configs #
####################
# Training and logging #
train_config = cfg.Config({
    'mode': 'rbp', #bptt, rbp, rbp-1, rbp-1h
    # 'verify_grad': False, #False, 'num', 'bptt'
    'batch_size': 64,
    'epochs': 1,
    'save_logs': True,
    'log_every': 10,
    'sparse_log_factor': 5,
    })

slurm_config = cfg.Config({
    'cpus': 1,
    'mem': 16, #GB
    'gpus': 0,
    'time': '0-12:00:00'
    })

# Networks #
base_net_config = cfg.Config({
    'class': None, #must define in specific config
    'converged_thres': 1e-6, #must be small or torch.allclose(FP_grad, BP_grad) fails
    'max_steps': 500,
    'dt': .05,
    # 'loss_fn': None,
    # 'acc_fn' : None,
    })


large_assoc_mem_config = deepcopy(base_net_config)
large_assoc_mem_config.update({
    'class': 'LargeAssociativeMemory',
    'input_size': None, #if None, infer from dataset
    'hidden_size': 25,
    'input_nonlin': networks.Identity(),
    'hidden_nonlin': networks.Softmax(beta=5, train=False),
    'tau': 1.,
    })


conv_net_config = deepcopy(base_net_config)
conv_net_config.update({
    'class': 'ConvThreeLayer',
    'x_size' : None, #if None, infer from dataset
    'x_channels' : None, #if None, infer from dataset
    'y_channels' : 50,
    'kernel_size' : 8,
    'stride' : 1,
    'z_size' : 50,
    })


# Datasets #
base_data_config = cfg.Config({
    'perturb_entries': 0.5,
    'perturb_mask': 'rand', #rand, first, last
    'perturb_value': 'min', #min, max, rand, <float>
    })


mnist_config = deepcopy(base_data_config)
mnist_config.update({
    'class':'AssociativeMNIST',
    'include_test': False,
    'num_samples': None, #if None takes entire dataset
    'select_classes': 'all',
    'n_per_class': 'all',
    'crop': False,
    'downsample': False,
    'normalize': False
    })


mnist_classify_config = deepcopy(base_data_config)
mnist_classify_config.update({
    'class':'AssociativeClassifyMNIST'
    })
del mnist_classify_config['perturb_mask']
del mnist_classify_config['perturb_entries']


cifar_config = deepcopy(base_data_config)
cifar_config.update({
    'class':'AssociativeCIFAR10',
    'include_test': False,
    'num_samples': None,
    'perturb_mask':'rand'
    })


###############
# Experiments #
###############
class Experiment:
    baseconfig = None
    deltaconfigs = {}
    deltamode = 'combinatorial' #combinatorial or sequential

    def __init__(self):
        self.configs, self.labels = cfg.flatten_config_loop(self.baseconfig, self.deltaconfigs, self.deltamode)

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx):
        return self.configs[idx], self.labels[idx]


class DefaultExperiment(Experiment):
    baseconfig = cfg.Config({ #be careful with in-place ops!
        'train': deepcopy(train_config),
        'net': deepcopy(large_assoc_mem_config),
        'data': deepcopy(mnist_classify_config),
        })


class Stepsize_Onestep_Beta_Convergence(Experiment):
    from networks import Softmax

    _train_config = deepcopy(train_config)
    _train_config.update({
        'mode': 'bptt', #bptt, rbp, rbp-1, rbp-1h
        'batch_size': 100,
        'epochs': 1000,
        'log_every': 50,
        'sparse_log_factor': 5,
        })

    _data_config = deepcopy(mnist_config)
    _data_config.update({
        'select_classes': [0,1,2,3,4],
        'n_per_class': 'all',
        })

    _net_config = deepcopy(large_assoc_mem_config)
    _net_config.update({
        'converged_thres': 1e-6,
        'max_steps': 500,
        'dt': .05,
        'hidden_size': 100,
        'input_nonlin': networks.Identity(),
        'hidden_nonlin': networks.Softmax(beta=1, train=False),
        'tau': 1.,
        })

    baseconfig = cfg.Config({
        'train': _train_config,
        'net': _net_config,
        'data': _data_config,
        'slurm': deepcopy(slurm_config)
        })

    deltaconfigs = {'net.dt': [0.1, 1],
                    'net.max_steps': [1000, 1],
                    'data.n_per_class': [20, 'all'],
                    'net.hidden_nonlin': [Softmax(1), Softmax(5), Softmax(10)]
                    }
    deltamode = 'combinatorial'


class Associative_CIFAR10_Debug(Experiment):
    _train_config = deepcopy(train_config)
    _train_config.update({
        'mode': 'bptt', #bptt, rbp, rbp-1, rbp-1h
        'batch_size': 1,
        'epochs': 5000,
        'save_logs': True,
        'log_every': 20,
        'sparse_log_factor': 5,
        })

    _net_config = deepcopy(conv_net_config)
    _net_config.update({
        'y_channels' : 4,
        'kernel_size' : 16,
        'stride' : 16,
        'z_size' : 50,
        })

    _data_config = deepcopy(cifar_config)
    _data_config.update({
        'num_samples': _train_config['batch_size'],
        'perturb_entries': 0.2
        })

    baseconfig = cfg.Config({
        'train': _train_config,
        'net': _net_config,
        'data': _data_config,
        })

    deltaconfigs = {'net.kernel_size': [4,4, 8,8,8, 16,16,16,16],
                    'net.stride':      [1,4, 1,4,8,  1, 4, 8,16]}
    deltamode = 'sequential'
