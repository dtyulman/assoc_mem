#built-in
import os
from copy import deepcopy

#third-party
import torch, joblib
import matplotlib.pyplot as plt

#for my machine only TODO: remove
try:
    os.chdir('/Users/danil/My/School/Columbia/Research/assoc_mem')
except FileNotFoundError as e:
    print(e.args)

#custom
import utils, data, training, networks
import configs as cfg
#%%
train_config = cfg.Config({
    'class': 'SGDTrain', #FPTrain, SGDTrain
    'batch_size': 256,
    'verify_grad': False,

    'optim': {'class': 'Adam', #CustomOpt, Adam
              # 'lr': 1e-3,

              #Adam only
              # 'weight_decay': 0,
              # 'amsgrad': False,

              #for CustomOpt only
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
    'acc_fn' : 'mae', #cls, L0, L1/mae
    'reg_fn': None, #L2, None
    'reg_rate': None,

    'epochs': 1000,
    'print_every': 10,
    'sparse_log_factor': 1,
    'device': 'cuda', #cuda, cpu
    })

net_config = cfg.Config({
    'class': 'ModernHopfield',
    'input_size': None, #if None, infer from dataset
    'hidden_size': 225,
    'init': 'random', #random, data_mean, inputs, targets
    'normalize_weight': True,
    'dropout': False, #float in (0,1), or False
    'beta': 7.,
    'train_beta': True,
    'tau': 1.,
    'normalize_input': False,
    'input_mode': 'clamp', #init, cont, init+cont, clamp
    'dt': .1,
    'num_steps': 50000,
    'fp_mode': 'iter', #iter, del2
    'fp_thres':1e-9,
    })


data_values_config = cfg.Config({
    'class': 'MNISTDataset', #MNISTDataset, RandomDataset
    'include_test': True,
    'normalize' : 'data', #data, data+targets, False
    'num_samples': None, #if None takes entire MNISTDataset, requires int for RandomDataset
    'balanced': False, #only for MNISTDataset or RandomDataset+'bern'

    #MNISTDataset only
    'crop': False,
    'downsample': False,

    #RandomDataset only
    'distribution': 'bern', #bern, unif, gaus
    'input_size': 30,
    'num_classes': 3,
    })


data_mode_config = cfg.Config({
    'classify': False,
    'perturb_entries': 0.5,
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
deltaconfigs = {'train.batch_size':[256,128], 'net.beta':[7., 9., 11., 13.]}
configs, labels = cfg.flatten_config_loop(baseconfig, deltaconfigs)
saveroot = utils.initialize_savedir(baseconfig)

for config, label in zip(configs, labels):
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
    device = utils.choose_device(config['train'].pop('device'))
    epochs = config['train'].pop('epochs')
    batch_size = config['train'].pop('batch_size')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if test_data:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=50)
    else:
        # if batch_size < len(train_data): #TODO: OOMs if using entire MNIST
        #     test_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))
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
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=optim_config.pop('lr', 1e-3),
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
    torch.save(net, os.path.join(savedir, 'net.pt'))

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
#             dnorm = ':nml' if config['data.normalize'] else '',
#             )
title = ''
net.to('cpu')
#%%
plots.plot_loss_acc(logger)

#%%
plots.plot_weights(net)


#%%
n_per_class = None
debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)
plots.plot_data_batch(debug_input, debug_target)

state_debug_history = net((debug_input, ~debug_perturb_mask), debug=True)
plots.plot_hidden_max_argmax(state_debug_history, n_per_class, apply_nonlin=True)

#%%
n_per_class = 1
debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)

num_steps_train = net.num_steps
# net.num_steps = 2000#int(100/net.dt)
# net.fp_thres = 1e-9
state_debug_history = net((debug_input, ~debug_perturb_mask), debug=True)

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
