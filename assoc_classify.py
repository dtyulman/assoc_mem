import importlib, datetime

#third-party
import torch, joblib

#custom
import plots, data, configs

#for my machine only TODO: remove
import os
if os.path.abspath('.').find('dtyulman') > -1:
    os.chdir('/home/dtyulman/projects/assoc_mem')

#??? Train on one step --> probably won't extrapolate to many
#??? Try "full" autoassociative task
#??? Can we remove the "dead" units?
#??? Normalize each weight vector per hidden unit to 1 --> removes the single giant hidden unit?

def run_config(config, savename=None, savedir='.'):
    #data
    if config['data']['name'] == 'MNIST':
        train_data, test_data = data.get_aa_mnist_classification_data(config['data']['include_test'])
    if config['data']['subset']:
        train_data.dataset.data = train_data.dataset.data[:50]
        train_data.dataset.targets = train_data.dataset.targets[:50]

    #network
    if config['net']['input_size'] is None:
        config['net']['input_size'] = train_data[0][0].numel()
    NetClass = getattr(importlib.import_module('networks'), config['net'].pop('class'))
    net = NetClass(**config['net'])

    #training
    epochs = config['train'].pop('epochs')
    B = config['train'].pop('batch_size')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=B, shuffle=True)
    if test_data:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    else:
        if B < len(train_data):
            test_loader = torch.utils.data.DataLoader(train_data, batch_size=B)
        else:
            test_loader = None

    TrainerClass = getattr(importlib.import_module('training'), config['train'].pop('class'))
    trainer = TrainerClass(net, train_loader, test_loader, **config['train'])

    #go
    logger = trainer(epochs, label=savename)
    if savename:
        joblib.dump(logger, os.path.join(savedir, f'log_{savename}.pkl'))
        torch.save(net, os.path.join(savedir, f'net_{savename}.pt'))
    return net, logger
#%%

# TODO: automatically make this folder structure
# assoc_mem/results/
# | -- yyyy-mm-dd/
# | | -- nnnn/
# | | | -- baseconfig.txt
# | | | -- deltaconfiglabel/
# | | | | -- net.pt
# | | | | -- log.pkl
# | | | | -- config.pkl

#get configs list
baseconfig = configs.Config({
    'train': {'class': 'FPTrain', #FPTrain, SGDTrain
              'batch_size': 50,
              'lr': 0.01, #for FPT only
              'print_every': 10,
              'loss_mode': 'full', # full, class
              'epochs':1000,
              },

    'net': {'class': 'ModernHopfield',
            'input_size': None, #if None, infer from dataset
            'hidden_size': 50,
            'beta': 100,
            'tau': 1,
            'input_mode': 'init', #init, cont, init+cont, clamp
            'dt': 0.05,
            'num_steps': 1000,
            'fp_mode': 'iter', #iter, del2
            'fp_thres':1e-9,
            },

    'data': {'name': 'MNIST',
             'subset': 50, #positive integer or False: takes only first N items
             'include_test': False,
             }
    }) #alternatively do configs.get_config() and modify defaults

configslist, labelslist = configs.flatten_config_loop(baseconfig,
                              {'net.input_mode': ['init', 'cont', 'init+cont'],
                               'train.loss_mode': ['full', 'class']})

#set up save dir
root = os.path.dirname(os.path.abspath(__file__))
ymd = datetime.date.today().strftime('%Y-%m-%d')
saveroot = os.path.join(root, 'results', ymd)
try:
    run_number = int(next(os.walk(saveroot))[1][-1])+1
except StopIteration:
    run_number = 0
savedir = os.path.join(saveroot, '{:04d}'.format(run_number))
os.makedirs(savedir)
with open(os.path.join(savedir, 'baseconfig.txt'), 'w') as f:
    f.write(repr(baseconfig))

#go
for config,label in zip(configslist, labelslist):
    net, logger = run_config(config, savename=label, savedir=savedir)



#%% plot
ax=None
ll = []
for log_fname in filter(lambda s: s.endswith('pkl'), next(os.walk('.'))[2]):
    logger = joblib.load(log_fname)
    label = log_fname[4:-4].replace('_', ', ')
    ax = plots.plot_loss_acc(logger['train_loss'], logger['train_acc'],
                  # logger['test_loss'], logger['test_acc'],
                  iters=logger['iter'],
                  title='ModernHopfield, MNIST_50, FPT',
                  ax=ax)
    ll.append(label)
ax[0].legend(ll)

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,3)
for i, net_fname in enumerate(filter(lambda s: s.endswith('.pt'), next(os.walk('.'))[2])):
    net = torch.load(net_fname)
    label = log_fname[4:-4].replace('_', ' ')
    net.to('cpu')

    a = plots.plot_weights_mnist(net.W, ax=ax.flatten()[i])
    a.set_title(label)

# # %%
# # n_per_class = 10
# # debug_data = data.AssociativeDataset(data.filter_classes(test_data.dataset,n_per_class=n_per_class))
# # debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))
# # state_debug_history = net(debug_input, debug=True)
# # plots.plot_hidden_max_argmax(state_debug_history, n_per_class)

# #%%
# import matplotlib.pyplot as plt
# net.to('cpu')

# n_per_class = 1
# debug_data = data.AssociativeDataset(data.filter_classes(train_data.dataset,n_per_class=n_per_class))
# debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))
# num_steps_train = net.num_steps
# net.num_steps = int(100/net.dt)
# net.fp_thres = 0
# state_debug_history = net(debug_input, debug=True)
# # plots.plot_state_update_magnitude_dynamics(state_debug_history, n_per_class, num_steps_train)

# # plots.plot_energy_dynamics(state_debug_history, net)

# fig, ax = plt.subplots(2,1, sharex=True)
# plots.plot_energy_dynamics(state_debug_history, net, num_steps_train=num_steps_train, ax=ax[0])
# plots.plot_max_hidden_dynamics(state_debug_history, num_steps_train=num_steps_train, ax=ax[1])
# fig.suptitle('FPT')
# ax[0].set_xlabel('')
# ax[0].legend_.remove()
# w,h = fig.get_size_inches()
# fig.set_size_inches(w,1.5*h)
# fig.tight_layout()
