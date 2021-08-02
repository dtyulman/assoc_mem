#built-in
import os
from copy import deepcopy

#third-party
import torch, joblib

#custom
import utils, data, training, networks
import configs as cfg

#for my machine only TODO: remove
if os.path.abspath('.').find('dtyulman') > -1:
    os.chdir('/home/dtyulman/projects/assoc_mem')

#%%
train_config = cfg.Config({
    'class': 'FPTrain', #FPTrain, SGDTrain
    'batch_size': 50,

    'lr': 0.0001,
    'lr_decay': 1.,
    'momentum': 0.99,
    'clip': False,
    'rescale_grad': True,
    'beta_decay': False, #1.00035,

    'loss_mode': 'full', #full, class
    'loss_fn': 'mse', #mse, cos, bce
    'acc_mode': 'class', #full, class
    'acc_fn' : 'cls', #cls, L0, L1/mae
    'reg_fn': None, #L2, None
    'reg_rate': None,

    'epochs': 500,
    'print_every': 10,
    'sparse_log_factor': 1,
    'device': 'cuda', #cuda, cpu
    })

net_config = cfg.Config({
    'class': 'ModernHopfield',
    'input_size': None, #if None, infer from dataset
    'hidden_size': 50,
    'normalize_weight': True,
    'dropout': False,
    'beta': 20.,
    'tau': 1.,
    'normalize_input': False,
    'input_mode': 'init', #init, cont, init+cont, clamp
    'dt': 0.1,
    'num_steps': 500,
    'fp_mode': 'iter', #iter, del2
    'fp_thres':1e-9,
    })

data_values_config = cfg.Config({
    'class': 'MNISTDataset', #MNISTDataset, RandomDataset
    'include_test': False,
    'normalize' : 'data+targets', #data, data+targets, False
    'num_samples': 50, #if None takes entire MNISTDataset
    'balanced': False, #only for MNISTDataset or RandomDataset+'bern'

    #MNISTDataset only
    'crop': False,
    'downsample': False,

    #RandomDataset only
    'distribution': 'bern', #bern, unif, gaus
    'input_size': 784,
    'num_classes': 10,
    })

data_mode_config = cfg.Config({
    'classify': True,
    'perturb_entries': 10,
    'perturb_mode': 'last',
    'perturb_value': 'min', #min, max, rand, <float>
    })


#be careful with in-place ops!
baseconfig = cfg.Config({
    'train': deepcopy(train_config),
    'net': deepcopy(net_config),
    'data': {'values': deepcopy(data_values_config),
             'mode': deepcopy(data_mode_config)},
    })


# baseconfig = cfg.load_config('./results/2021-07-27/proof_of_principle/baseconfig.txt')
# baseconfig['train.loss_mode'] = 'class'

#%%
# deltaconfigs = {'net.num_steps': [1, 1000],
#                 'train.loss_mode': ['full', 'class'],
#                 'net.input_mode': ['init', 'cont', 'init+cont'],
#                 }
deltaconfigs = {}
configs, labels = cfg.flatten_config_loop(baseconfig, deltaconfigs)
saveroot = utils.initialize_savedir(baseconfig)

for config, label in zip(configs, labels):
    cfg.verify_config(config)
    print(config)
    savedir = os.path.join(saveroot, label)

    #data
    train_data, test_data = data.get_aa_data(config['data.values'], config['data.mode'])

    #network
    if config['net']['input_size'] is None:
        config['net']['input_size'] = train_data[0][0].numel()
    NetClass = getattr(networks, config['net'].pop('class'))
    net = NetClass(**config['net'])
    # with torch.no_grad():
    #     net._W += train_data[:][0].mean(dim=0).tile(1,net.hidden_size).T
    #     net._maybe_normalize_weight()
    with torch.no_grad():
        net._W.data = deepcopy(train_data[:][1].squeeze())


    #training
    device = utils.choose_device(config['train'].pop('device'))
    epochs = config['train'].pop('epochs')
    batch_size = config['train'].pop('batch_size')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if test_data:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    else:
        if batch_size < len(train_data):
            test_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))
        else:
            test_loader = None

    reg_fn = config['train'].pop('reg_fn')
    if reg_fn == 'L2':
        reg_loss = training.L2Reg({'_W': net._W}, reg_rate=config['train'].pop('reg_rate'))
    elif reg_fn is None:
        reg_loss = None
    else:
        raise ValueError(f"Invalid reg_fn: '{reg_fn}'")

    TrainerClass = getattr(training, config['train'].pop('class'))
    trainer = TrainerClass(net, train_loader, test_loader, reg_loss=reg_loss, logdir=savedir,
                           **config['train'])

    #go
    net.to(device)
    logger = trainer(epochs, label=label)

    # joblib.dump(logger, os.path.join(savedir, 'log.pkl'))
    torch.save(net, os.path.join(savedir, 'net.pt'))

    print()


#%% plot
import plots
import matplotlib.pyplot as plt

#TODO: plot v-t averaged over B

# title = '{net}{hid} W:{norm} $\\beta$={bet} $\\tau$={tau} in:{inp}\n' \
#         '{trn} B={bat} lr={lr} L:{loss} MNIST{sub}{dnorm}'.format(
#             net = 'MH',
#             hid = net.hidden_size,
#             norm = "nml" if net.normalize_weight else 'raw',
#             bet = net.beta,
#             tau = net.tau,
#             inp = net.input_mode,
#             trn = trainer.name[:2],
#             bat = train_loader.batch_size,
#             lr = trainer.lr,
#             loss = trainer.loss_mode,
#             sub = f'{subset}' if subset else '',
#             dnorm = ':nml' if config['data']['normalize'] else '',
#             )
title = ''
net.to('cpu')
#%%
plots.plot_loss_acc(logger)

#%%
plot_class=True
if net.normalize_weight:
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    plots.plot_weights_mnist(net._W, plot_class=plot_class, ax=ax[0])
    ax[0].set_title('Raw')
    plots.plot_weights_mnist(net.W, plot_class=plot_class, ax=ax[1])
    ax[1].set_title('Normalized')
    plots.scale_fig(fig, 1.5)
else:
    ax = plots.plot_weights_mnist(net._W, plot_class=plot_class)
    ax.set_title('Raw (no normalization)')
    fig = ax.get_figure()
fig.tight_layout()


#%%
n_per_class = None
debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)
plots.plot_data_batch(debug_input, debug_target)

state_debug_history = net(debug_input, clamp_mask=~debug_perturb_mask, debug=True)
plots.plot_hidden_max_argmax(state_debug_history, n_per_class, apply_nonlin=True)

#%%
n_per_class = 1
debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)

num_steps_train = net.num_steps
# net.num_steps = 2000#int(100/net.dt)
# net.fp_thres = 1e-9
state_debug_history = net(debug_input, clamp_mask=~debug_perturb_mask, debug=True)

fig, ax = plt.subplots(2,2, sharex=True)
ax = ax.flatten()

plots.plot_state_update_magnitude_dynamics(state_debug_history, n_per_class, num_steps_train, ax=ax[0])
plots.plot_energy_dynamics(state_debug_history, net, num_steps_train=num_steps_train, ax=ax[1])
plots.plot_hidden_dynamics(state_debug_history, transformation='max', num_steps_train=num_steps_train, ax=ax[2])
plots.plot_hidden_dynamics(state_debug_history, apply_nonlin=False, transformation='mean', ax=ax[3])

[a.set_xlabel('') for a in ax[0:2]]
[a.legend_.remove() for a in ax[0:-1]]
fig.suptitle(title)
plots.scale_fig(fig, 1.5, 1.5)
fig.tight_layout()

#%%
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
plots.plot_state_dynamics(state_debug_history, ax=ax[0])
plots.plot_state_dynamics(state_debug_history, targets=debug_target, ax=ax[1]) #plot error instead of state
plots.scale_fig(fig, 1.6, 3.5)
fig.tight_layout()
