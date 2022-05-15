#built-in
import os
from copy import deepcopy

#third-party
import torch, joblib
from torch import nn
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5agg') #prevents figs from stealing focus during training

#for my machine only TODO: remove
try:
    os.chdir('/Users/danil/My/School/Columbia/Research/assoc_mem')
except FileNotFoundError as e:
    print(e.args)

#custom
import utils, data, training, networks, plots
import configs as cfg
#%%
train_config = cfg.Config({
    'class': 'FPTrain', #FPTrain, BPTrain
    'batch_size': 128,
    #FPTrain only
    'verify_grad': '', # '', 'num', 'bptt', 'num+bptt'
    'approx': False, #False, first, first-heuristic, inv-heuristic

    'optim': {'class': 'Adam', #CustomOpt, Adam
              'lr': .005,
              #Adam only
              'weight_decay': 1e-9,
              'amsgrad': False,
              #CustomOpt only
              'lr_decay': 1.,
              'momentum': 0.999, #float, False
              'clip': False, #norm, value, False
              'clip_thres': None, #ignored if clip==False
              'rescale_grad': False, #True, False
              'beta_increment': False, #int, False
              'beta_max': None, #ignored if beta_increment==False
              },

    'loss_mode': 'full', #full, class
    'loss_fn': 'mse', #mse, cos, bce
    'acc_mode': 'full', #full, class
    'acc_fn' : 'L1', #cls, L0, L1/mae
    'reg_fn': None, #L2, None
    'reg_rate': None,

    'epochs': 1,
    'print_every': 10,
    'sparse_log_factor': 5,
    'device': 'cuda', #cuda, cpu
    })

net_config = cfg.Config({
    'class': 'LargeAssociativeMem',
    'input_size': None, #if None, infer from dataset
    'hidden_size': 25,
    'init': 'random', #random, data_mean, inputs, targets
    'normalize_weight': True,
    'f': networks.Softmax(beta=0.1, train=True),
    'g': networks.Spherical(),
    'normalize_input': False, #normalizes post-perturbation but before showing it to network (data.normalize does it before, g=Spherical does it at every step)
    'input_mode': 'clamp', #init, cont, init+cont, clamp
    'tau': 1.,
    'dt': 0.1,
    'num_steps': 5000,
    'check_converged': True,
    'fp_mode': 'iter', #iter, del2
    'fp_thres': 1e-6, #must be small or torch.allclose(FP_grad, BP_grad) fails
    'dropout': False, #float in (0,1), or False
    })


data_values_config = cfg.Config({
    'class': 'MNISTDataset', #MNISTDataset, RandomDataset
    'include_test': False,
    'normalize' : 'data', #data, data+targets, False
    'num_samples': 25, #if None takes entire MNISTDataset, requires int for RandomDataset
    'balanced': False, #only for MNISTDataset or RandomDataset+'bern'

    #MNISTDataset only
    'crop': False,
    'downsample': False,

    #RandomDataset only
    # 'distribution': 'bern', #bern, unif, gaus
    # 'input_size': 30,
    # 'num_classes': 3,
    })


data_mode_config = cfg.Config({
    'classify': False,
    'perturb_entries': 0.5,
    'perturb_mode': 'rand', #rand, first, last (note: rand mask same for whole batch)
    'perturb_value': 'min', #min, max, rand, <float>
    })


#be careful with in-place ops!
baseconfig = cfg.Config({
    'train': deepcopy(train_config),
    'net': deepcopy(net_config),
    'data': {'values': deepcopy(data_values_config),
             'mode': deepcopy(data_mode_config)},
    })


deltaconfigs = {}
mode = 'combinatorial'

#%%
#mem
# baseconfig['net.hidden_size'] = 25
# baseconfig['train.batch_size'] = 25
# baseconfig['train.epochs'] = 5000
# baseconfig['train.print_every'] = 10
# baseconfig['data.values.num_samples'] = 25
# baseconfig['data.values.include_test'] = False

#small
baseconfig['net.hidden_size'] = 25
baseconfig['train.batch_size'] = 128
baseconfig['train.epochs'] = 100
baseconfig['train.print_every'] = 100
baseconfig['data.values.num_samples'] = None
baseconfig['data.values.include_test'] = True

#big
# baseconfig['net.hidden_size'] = 500
# baseconfig['train.batch_size'] = 64
# baseconfig['train.epochs'] = 25
# baseconfig['train.print_every'] = 100
# baseconfig['data.values.num_samples'] = None
# baseconfig['data.values.include_test'] = True


deltas = [# train      approx             nonlin
           ['FPTrain',  'first',           networks.Softmax(beta=.1,train=True)],
           ['FPTrain',  'first-heuristic', networks.Softmax(beta=.1, train=True)],
           ['FPTrain',  False,             networks.Softmax(beta=.1, train=True)],
           ['BPTrain',  None,              networks.Softmax(beta=.1, train=True)],
]

deltaconfigs = {'train.class': [delta[0] for delta in deltas],
                'train.approx': [delta[1] for delta in deltas],
                'net.f': [delta[2] for delta in deltas],
                'net.normalize_weight': [delta[3] for delta in deltas]
                }
mode = 'sequential'

#%%
configs, labels = cfg.flatten_config_loop(baseconfig, deltaconfigs, mode=mode)
saveroot = utils.initialize_savedir(baseconfig)
device = None
for config, label in zip(configs, labels):
    torch.random.manual_seed(3)

    cfg.verify_config(config)
    config_copy = deepcopy(config)
    print(config)
    savedir = os.path.join(saveroot, label)

    if config['train.verify_grad']:
        torch.set_default_dtype(torch.float64) #double precision for numerical grad approx
    else:
        torch.set_default_dtype(torch.float) #pytorch's default b/c faster for GPUs

    #data
    train_data, test_data = data.get_aa_data(config['data.values'], config['data.mode'])

    #network
    if config['net.input_size'] is None:
        config['net.input_size'] = train_data[0][0].numel()
    init = config['net'].pop('init')
    NetClass = getattr(networks, config['net'].pop('class'))
    net = NetClass(**config['net'])
    with torch.no_grad():
        if init == 'random':
            # mean = 0.01
            # std = 0.001
            # print(f'OVERRIDING DEFAULT RANDOM INIT, mean={mean}, std={std}')
            # net._W.data = torch.randn_like(net._W )*std + mean
            pass #happens by default inside the net class
        elif init == 'data_mean':
            net._W = train_data[:][0].mean(dim=0).tile(1,net.hidden_size).T
            #add small noise to prevent identical hidden units
            net._W += torch.randn_like(net._W)*net._W.std()/10
        elif init == 'inputs':
            net._W.data = train_data[:net.hidden_size][0].squeeze()
        elif init == 'targets':
            net._W.data = train_data[:net.hidden_size][1].squeeze()
        else:
            raise ValueError(f"Invalid network init: '{init}'")
        net._maybe_normalize_weight()


    #training
    if device is None:
        device = utils.choose_device(config['train'].pop('device'))
    epochs = config['train'].pop('epochs')
    batch_size = config['train'].pop('batch_size')

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if test_data:
        test_loader = data.DataLoader(test_data, batch_size=batch_size)
    else:
        # if batch_size < len(train_data): #TODO: OOMs if using entire MNIST
        #     test_loader = data.DataLoader(train_data, batch_size=len(train_data))
        # else:
        test_loader = None

    reg_fn = config['train'].pop('reg_fn')
    reg_rate = config['train'].pop('reg_rate')
    if reg_fn == 'L2':
        reg_loss = training.L2Reg({'_W': net._W}, reg_rate)
    elif reg_fn is None:
        reg_loss = None
    else:
        raise ValueError(f"Invalid reg_fn: '{reg_fn}'")

    optim_config = config['train'].pop('optim')
    optim_class = optim_config.pop('class')
    if optim_class == 'Adam':
        lr = optim_config.pop('lr', 1e-3)
        if 'beta' in net.named_parameters():
            params_except_beta = dict(net.named_parameters())
            del params_except_beta['f.beta']
            params_except_beta = params_except_beta.values()
            optimizer = torch.optim.Adam([{'params': params_except_beta},
                                          {'params': net.f.beta, 'lr':lr/10.}],
                                         lr=lr,
                                         weight_decay=optim_config.pop('weight_decay', 0),
                                         amsgrad=optim_config.pop('amsgrad', False))
        else:
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay=optim_config.pop('weight_decay', 0),
                                         amsgrad=optim_config.pop('amsgrad', False))
    elif optim_class == 'CustomOpt':
        optimizer = training.CustomOpt(net, **optim_config)

    TrainerClass = getattr(training, config['train'].pop('class'))
    trainer = TrainerClass(net, train_loader, test_loader, optimizer=optimizer, reg_loss=reg_loss,
                           logdir=savedir, **config['train'])

    #go
    net.to(device)
    logger = trainer(epochs, label=label)
    logger.add_text('config', str(config_copy))

    # joblib.dump(logger, os.path.join(savedir, 'log.pkl'))
    net.to('cpu')
    torch.save(net, os.path.join(savedir, 'net.pt'))

# %% plot
# plots.plot_loss_acc(logger)

# #%%
# plots.plot_weights(net, drop_last=0)

# #%%
# n_per_class = 10
# debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)
# plots.plot_data_batch(debug_input, debug_target)

# state_debug_history = net((debug_input, ~debug_perturb_mask), debug=True)
# plots.plot_hidden_max_argmax(state_debug_history, n_per_class, apply_nonlin=True)

# #%%
# n_per_class = 1
# debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)

# num_steps = net.num_steps
# # fp_thres = net.fp_thres
# # check_converged = net.check_converged
# # net.num_steps = 1e10#int(100/net.dt)
# # net.fp_thres = 1e-9
# # net.check_converged = True

# state_debug_history = net((debug_input, ~debug_perturb_mask), debug=True)

# # net.num_steps = num_steps
# # net.fp_thres = fp_thres
# # net.check_converged = check_converged

# fig, ax = plt.subplots(2,2, sharex=True)
# ax = ax.flatten()

# plots.plot_state_update_magnitude_dynamics(state_debug_history, n_per_class, num_steps, ax=ax[0])
# plots.plot_energy_dynamics(state_debug_history, net, num_steps_train=num_steps, ax=ax[1])
# plots.plot_hidden_dynamics(state_debug_history, transformation='max', num_steps_train=num_steps, ax=ax[2])
# plots.plot_hidden_dynamics(state_debug_history, apply_nonlin=False, transformation='mean', ax=ax[3])

# [a.set_xlabel('') for a in ax[0:2]]
# [a.legend_.remove() for a in ax[0:-1]]
# plots.scale_fig(fig, 1.5, 1.5)
# fig.tight_layout()

# #%%
# fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
# plots.plot_state_dynamics(state_debug_history, ax=ax[0])
# plots.plot_state_dynamics(state_debug_history, targets=debug_target, ax=ax[1]) #plot error instead of state
# plots.scale_fig(fig, 1.6, 3.5)
# fig.tight_layout()

# #%%
# # beta = net.f.beta.item()
# # net.f.beta.data = torch.tensor(9.)
# plots.plot_fixed_points(net, num_fps=100, drop_last=0)
# # net.f.beta.data = torch.tensor(beta)
