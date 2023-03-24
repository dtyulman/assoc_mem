from copy import deepcopy

import numpy as np

import networks
import data
import components as nc
import configs as cfg

####################
# Pre-made configs #
####################
# Training and logging #
train_config = cfg.Config({
    'mode': 'bptt', #bptt, rbp, rbp-1, rbp-1h
    # 'verify_grad': False, #False, 'num', 'bptt'
    'batch_size': 64,
    'max_epochs': 1,
    'max_iters': -1,
    'save_logs': True,
    'log_every': 10,
    'sparse_log_factor': 5,
    # 'trainer_kwargs': cfg.Config({}) #any kwargs passed to pl.Trainer()
    })

slurm_config = cfg.Config({
    'cpus': 1,
    'mem': 16, #GB
    'gpus': 0,
    'time': '0-12:00:00'
    })

# Networks #
base_net_config = cfg.Config({
    'class': None, #must define in sub-config
    'converged_thres': 1e-6, #must be small or torch.allclose(FP_grad, BP_grad) fails
    'max_steps': 500,
    'dt': .05,
    'input_mode': 'init' #init, clamp
    # 'loss_fn': None,
    # 'acc_fn' : None,
    })


large_assoc_mem_config = deepcopy(base_net_config)
large_assoc_mem_config.update({
    'class': networks.LargeAssociativeMemory,
    'visible_size': None, #if None, infer from dataset
    'hidden_size': 25,
    'visible_nonlin': nc.Identity(),
    'hidden_nonlin': nc.Softmax(beta=1, train=True),
    'rescale_grads': False, #True, False
    'normalize_weights': False,
    'tau': 1.,
    })


exceptions_mhn_config = deepcopy(large_assoc_mem_config)
del exceptions_mhn_config['hidden_nonlin'] #hidden is hard-coded Softmax
exceptions_mhn_config.update({
    'class': networks.ExceptionsMHN,
    'beta': 1,
    'beta_exception': None, #if None, will use same beta for all inputs
    'train_beta': True,
    'exception_loss_scaling': 1, #int, 'linear_data'
    'exception_loss_mode': 'manual', #manual, entropy, max, norm, time
    'rescale_grads': False, #True, False, 'unweighted_loss'
})


conv_net_config = deepcopy(base_net_config)
conv_net_config.update({
    'class': networks.ConvThreeLayer,
    'x_size' : None, #if None, infer from dataset
    'x_channels' : None, #if None, infer from dataset
    'y_channels' : 50,
    'kernel_size' : 8,
    'stride' : 1,
    'z_size' : 50,
    })


# Datasets #
base_data_config = cfg.Config({
    'class': None, #must specify in sub-config
    'perturb_entries': 0.5,
    'perturb_mask': 'rand', #rand, first, last
    'perturb_value': 'min', #min, max, rand, <float>
    })


mnist_config = deepcopy(base_data_config)
mnist_config.update({
    'class': data.AssociativeMNIST,
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
    'class': data.AssociativeClassifyMNIST
    })
del mnist_classify_config['perturb_mask']
del mnist_classify_config['perturb_entries']


cifar_config = deepcopy(base_data_config)
cifar_config.update({
    'class': data.AssociativeCIFAR10,
    'include_test': False,
    'num_samples': None,
    'perturb_mask':'rand'
    })


exceptions_dataset_config = cfg.Config({
    'class': data.ExceptionsDataset,
    'dataset_config': deepcopy(mnist_config),
    'exceptions': [1],
    })
exceptions_dataset_config['dataset_config']['select_classes'] = [0, 1]
exceptions_dataset_config['dataset_config']['n_per_class'] = ['all', 3]


###############
# Experiments #
###############
class Experiment:
    baseconfig = None
    deltaconfigs = {}
    deltamode = 'combinatorial' #combinatorial or sequential

    def __init__(self):
        self.configs, self.labels = cfg.flatten_config_loop(
            self.baseconfig, self.deltaconfigs, self.deltamode)

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx):
        return self.configs[idx], self.labels[idx]



class DefaultExperiment(Experiment):
    baseconfig = cfg.Config({ #be careful with in-place ops!
        'train': deepcopy(train_config),
        'net': deepcopy(large_assoc_mem_config),
        'data': deepcopy(mnist_config),
        # 'slurm': deepcopy(slurm_config)
        })



class AssociativeMNIST_Exceptions(Experiment):
    from components import Softmax, Identity, Spherical

    _train_config = deepcopy(train_config)
    _train_config.update({
        'batch_size': 100,
        'max_epochs': None,
        'max_iters': -1,
        'log_every': 50,
        'sparse_log_factor': 5,
        })

    _data_config = deepcopy(exceptions_dataset_config)
    _data_config['dataset_config']['normalize'] = True

    _net_config = deepcopy(exceptions_mhn_config)
    _net_config.update({
        'visible_nonlin': Spherical(),
        'normalize_weights': 'rows',
        'beta_exception': None,
        'converged_thres': 1e-6,
        'max_steps': 1000,
        'input_mode': 'clamp',
        'hidden_size': 100,
        })

    baseconfig = cfg.Config({
        'train': _train_config,
        'net': _net_config,
        'data': _data_config,
        })

    deltaconfigs = {'net.beta': [1, 10],
                    'net.train_beta': [False, True],
                    'net.rescale_grads': [True, False, 'unweighted_loss'],
                    'net.exception_loss_scaling': [2000, 100, 10, 1],
                    }



class AssociativeMNIST_Exceptions_Automatic(AssociativeMNIST_Exceptions):
    from components import Softmax, Identity, Spherical
    baseconfig = deepcopy(AssociativeMNIST_Exceptions.baseconfig)
    baseconfig.update({'net.train_beta': False})

    deltaconfigs = {'net.beta': [1, 10],
                    'net.exception_loss_scaling': [0, 1, 10, 50, 100],
                    'net.exception_loss_mode': ['entropy', 'max', 'norm', 'time']#, 'manual'],
                    }



class AssociativeMNIST_Exceptions_TrainBeta(AssociativeMNIST_Exceptions):
    from components import Softmax, Identity, Spherical
    baseconfig = deepcopy(AssociativeMNIST_Exceptions.baseconfig)
    baseconfig.update({'net.train_beta': True})

    deltaconfigs = {'net.beta': np.linspace(0.1, 20, 20),
                    'net.rescale_grads': [False],
                    'net.exception_loss_scaling': [100, 10, 1],
                    }


class AssociativeMNIST_Exceptions_TwoBeta(AssociativeMNIST_Exceptions):
    from components import Softmax, Identity, Spherical
    baseconfig = deepcopy(AssociativeMNIST_Exceptions.baseconfig)
    baseconfig.update({'net.train_beta': False})

    deltaconfigs = {'net.beta': [0.1, 1],
                    'net.beta_exception': [1, 10],
                    'net.rescale_grads': [True, False],
                    'net.exception_loss_scaling': [2000, 100, 10, 1],
                    }


class AssociativeMNIST_Baseline_Clamped_Normalized(Experiment):
    from components import Softmax, Identity, Spherical

    _train_config = deepcopy(train_config)
    _train_config.update({
        'batch_size': 100,
        'max_epochs': None,
        'max_iters': 300*1000,
        'log_every': 50,
        'sparse_log_factor': 5,
        })

    _data_config = deepcopy(mnist_config)
    _data_config.update({
        'n_per_class': 10,
        'normalize': True
        })

    _net_config = deepcopy(large_assoc_mem_config)
    _net_config.update({
        'converged_thres': 1e-6,
        'max_steps': 1000,
        'dt': .05,
        'input_mode': 'clamp',
        'hidden_size': 100,
        'visible_nonlin': Spherical(),
        'hidden_nonlin': Softmax(beta=1, train=False),
        'tau': 1.,
        })

    baseconfig = cfg.Config({
        'train': _train_config,
        'net': _net_config,
        'data': _data_config,
        })

    deltaconfigs = {'net.visible_nonlin': [Identity(), Spherical()],
                    'net.hidden_nonlin': [Softmax(0.1, train=True),
                                          Softmax(1, train=True),
                                          Softmax(10, train=True),
                                          Softmax(0.1),
                                          Softmax(1),
                                          Softmax(10)]
                    }





class AssociativeMNIST_Baseline_Clamped_Normalized_NormalizedWeight(
        AssociativeMNIST_Baseline_Clamped_Normalized):
    from components import Softmax, Identity, Spherical
    baseconfig = deepcopy(AssociativeMNIST_Baseline_Clamped_Normalized.baseconfig)
    deltaconfigs = {'net.visible_nonlin': [Identity(), Spherical()],
                    'net.hidden_nonlin': [Softmax(0.1, train=True),
                                          Softmax(1, train=True),
                                          Softmax(10, train=True),
                                          Softmax(0.1),
                                          Softmax(1),
                                          Softmax(10)],
                    'net.normalize_weights': ['frobenius', 'rows', 'rows_scaled'],
                    'net.rescale_grads': [False, True]
                    }



class AssociativeMNIST_Baseline_RectPoly_Normalized(AssociativeMNIST_Baseline_Clamped_Normalized):
    from components import Softmax, Identity, Spherical, RectifiedPoly

    deltaconfigs = {'net.input_mode': ['init', 'clamp'],
                    'net.visible_nonlin': [Identity(), Spherical()],
                    'net.hidden_nonlin': [RectifiedPoly(n=2),
                                          RectifiedPoly(n=3),
                                          RectifiedPoly(n=5),
                                          RectifiedPoly(n=10),
                                          RectifiedPoly(n=2, train=True),
                                          RectifiedPoly(n=3, train=True),
                                          RectifiedPoly(n=5, train=True),
                                          RectifiedPoly(n=10, train=True)],
                    }


class AssociativeMNIST_Baseline_RectPoly(AssociativeMNIST_Baseline_RectPoly_Normalized):
    from components import Softmax, Identity, Spherical, RectifiedPoly

    baseconfig = deepcopy(AssociativeMNIST_Baseline_RectPoly_Normalized.baseconfig)
    baseconfig.update({'data.normalize': False})

    deltaconfigs = {'net.input_mode': ['init', 'clamp'],
                    'net.visible_nonlin': [Identity()],
                    'net.hidden_nonlin': [RectifiedPoly(n=2),
                                          # RectifiedPoly(n=3),
                                          # RectifiedPoly(n=5),
                                          # RectifiedPoly(n=10),
                                          # RectifiedPoly(n=2, train=True),
                                          # RectifiedPoly(n=3, train=True),
                                          # RectifiedPoly(n=5, train=True),
                                          # RectifiedPoly(n=10, train=True)
                                          ],
                    }


class Stepsize_Onestep_Beta_Convergence(Experiment):
    from components import Softmax

    _train_config = deepcopy(train_config)
    _train_config.update({
        'batch_size': 100,
        'max_epochs': None,
        'max_iters': 300*1000,
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
        'hidden_nonlin': nc.Softmax(beta=1, train=False),
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
