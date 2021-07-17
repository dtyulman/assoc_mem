import os

#third-party
import torch, joblib

import utils, data, training, networks
import configs as cfg

#for my machine only TODO: remove
if os.path.abspath('.').find('dtyulman') > -1:
    os.chdir('/home/dtyulman/projects/assoc_mem')

#%%
#get configs list
baseconfig = cfg.Config({
    'train': {'class': 'FPTrain', #FPTrain, SGDTrain
              'batch_size': 50,
              'lr': .1,
              'lr_decay': 1.,
              'print_every': 10,
              'loss_mode': 'full', #full, class
              'loss_fn': 'mse', #mse, cos, bce
              'acc_mode': 'full', #full, class
              'acc_fn' : 'mae', #mae, L0
              'epochs': 5000,
              'device': 'cuda', #cuda, cpu
              },

    'net': {'class': 'ModernHopfield',
            'input_size': None, #if None, infer from dataset
            'hidden_size': 50,
            'normalize_weight': True,
            'beta': 100,
            'tau': 1,
            'normalize_input': False,
            'input_mode': 'clamp', #init, cont, init+cont, clamp
            'dt': 0.05,
            'num_steps': 1000,
            'fp_mode': 'iter', #iter, del2
            'fp_thres':1e-9,
            },

    'data': {'name': 'MNIST',
             'include_test': False,
             'normalize' : True,
             'balanced': False,
             'crop': False,
             'downsample': False,
             'subset': 50, #if int, takes only first N items

             # 'mode': 'classify', #classify, complete
             # 'perturb_frac': None,
             # 'perturb_num': 10,
             # 'perturb_mode': 'last',
             # 'perturb_value': 0,

             'mode': 'complete', #classify, complete
             'perturb_frac': 0.5,
             'perturb_num': None,
             'perturb_mode': 'last',
             'perturb_value': 0,
             }
    }) #alternatively can do cfg.get_config() and modify defaults, be careful with in-place ops

# deltaconfigs = {'net.num_steps': [1, 1000],
#                 'train.loss_mode': ['full', 'class'],
#                 'net.input_mode': ['init', 'cont', 'init+cont'],
#                 }
deltaconfigs = {}
configs, labels = cfg.flatten_config_loop(baseconfig, deltaconfigs)

#%%
save = False
if save:
    savedir = utils.initialize_savedir(baseconfig)

for config, label in zip(configs, labels):
    cfg.verify_config(config)

    #data
    subset = config['data'].pop('subset')
    if config['data'].pop('name') == 'MNIST':
        aa_kwargs = {k : config['data'][k] for k in ['mode', 'perturb_frac', 'perturb_num', 'perturb_mode', 'perturb_value']}
        mnist_kwargs = {k : config['data'][k] for k in ['include_test','normalize', 'balanced', 'crop', 'downsample']}
        train_data, test_data = data.get_aa_mnist_data(mnist_kwargs, aa_kwargs)
    if subset:
        train_data.dataset.data = train_data.dataset.data[:50]
        train_data.dataset.targets = train_data.dataset.targets[:50]

    #network
    if config['net']['input_size'] is None:
        config['net']['input_size'] = train_data[0][0].numel()
    NetClass = getattr(networks, config['net'].pop('class'))
    net = NetClass(**config['net'])

    #training
    device = config['train'].pop('device')
    epochs = config['train'].pop('epochs')
    batch_size = config['train'].pop('batch_size')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if test_data:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    else:
        if batch_size < len(train_data):
            test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        else:
            test_loader = None
    TrainerClass = getattr(training, config['train'].pop('class'))
    trainer = TrainerClass(net, train_loader, test_loader, **config['train'])

    #go
    net.to(device)
    logger = trainer(epochs, label=label)

    if save:
        joblib.dump(logger, os.path.join(savedir, f'log_{label}.pkl'))
        torch.save(net, os.path.join(savedir, f'net_{label}.pt'))

    print()


#%% plot
import plots
import matplotlib.pyplot as plt

#TODO: plot v-t averaged over B

title = '{net}:{hid} W:{norm} $\\beta$={bet} $\\tau$={tau} in:{inp}\n' \
        '{trn} B={bat} lr={lr} L:{loss} MNIST{sub}'.format(
            net = 'MH',
            hid = net.hidden_size,
            norm = "nml" if net.normalize_weight else 'raw',
            bet = net.beta,
            tau = net.tau,
            inp = net.input_mode,
            trn = trainer.name[:2],
            bat = train_loader.batch_size,
            lr = trainer.lr,
            loss = trainer.loss_mode,
            sub = f':{subset}' if subset else '')

net.to('cpu')
#%%
plots.plot_loss_acc(logger['train_loss'], logger['train_acc'], iters=logger['iter'])

#%%
plot_class=True
if net.normalize_weight:
    fig, ax = plt.subplots(1,2)
    plots.plot_weights_mnist(net._W, plot_class=plot_class, ax=ax[0])
    ax[0].set_title('Raw')
    plots.plot_weights_mnist(net.W, plot_class=plot_class, ax=ax[1])
    ax[1].set_title('Normalized')
    w,h = fig.get_size_inches()
    fig.set_size_inches(w*1.5, h)
else:
    ax = plots.plot_weights_mnist(net._W, plot_class=plot_class)
    ax.set_title('Raw (no normalization)')
    fig = ax.get_figure()
fig.tight_layout()


#%%
n_per_class = 5
debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data)
plots.plot_data_batch(debug_input, debug_target)

state_debug_history = net(debug_input, debug=True)
plots.plot_hidden_max_argmax(state_debug_history, n_per_class, apply_nonlin=True)

#%%
n_per_class = 1
debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)
if net.input_mode == 'clamp':
    net.clamp_mask = ~debug_perturb_mask

num_steps_train = net.num_steps
net.num_steps = 200#int(100/net.dt)
net.fp_thres = 0
state_debug_history = net(debug_input, debug=True)

fig, ax = plt.subplots(2,2, sharex=True)
ax = ax.flatten()

plots.plot_state_update_magnitude_dynamics(state_debug_history, n_per_class, num_steps_train, ax=ax[0])
plots.plot_energy_dynamics(state_debug_history, net, num_steps_train=num_steps_train, ax=ax[1])
plots.plot_hidden_dynamics(state_debug_history, transformation='max', num_steps_train=num_steps_train, ax=ax[2])
plots.plot_hidden_dynamics(state_debug_history, apply_nonlin=False, transformation='mean', ax=ax[3])

[a.set_xlabel('') for a in ax[0:2]]
[a.legend_.remove() for a in ax[0:-1]]
fig.suptitle(title)
w,h = fig.get_size_inches()
fig.set_size_inches(1.5*w,1.5*h)
fig.tight_layout()

#%%
plots.plot_state_dynamics(state_debug_history)
