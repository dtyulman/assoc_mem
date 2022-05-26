import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import torch

import data, plots
import configs as cfg

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
sns.set(font='Arial',
        font_scale=7/12., #default size is 12pt, scale down to 7pt
        palette='Set1',
        rc={'axes.axisbelow': True,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'text.color': 'dimgrey', #e.g. legend

            'lines.solid_capstyle': 'round',
            'legend.facecolor': 'white',
            'legend.framealpha':0.8,

            'xtick.bottom': True,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',

            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': True,

             'xtick.major.size': 2,
             'xtick.major.width': .5,
             'xtick.minor.size': 1,
             'xtick.minor.width': .5,

             'ytick.major.size': 2,
             'ytick.major.width': .5,
             'ytick.minor.size': 1,
             'ytick.minor.width': .5
            }
        )

def get_data(logdir, key):
    ea = EventAccumulator(logdir)
    ea.Reload()
    walltimes, steps, vals = zip(*ea.Scalars(key))
    return np.array(walltimes), np.array(steps), np.array(vals)

root = '/Users/danil/My/School/Columbia/Research/assoc_mem/results/rbp_validation_1/'
scheme_lookup = {'Memory': 'mem',
                 'Small': 'small',
                 'Large': 'big'}
alg_lookup = {'BPTT': 'class=BPTrain approx=None f=Softmax(beta=0.1, train=True)',
              'RBP': 'class=FPTrain approx=False f=Softmax(beta=0.1, train=True)',
              'RBP-1': 'class=FPTrain approx=first f=Softmax(beta=0.1, train=True)',
              'RBP-1H': 'class=FPTrain approx=first-heuristic f=Softmax(beta=0.1, train=True)'}

# root = '/Users/danil/My/School/Columbia/Research/assoc_mem/results/rbp_validation_1/normalize_weight/'
# alg_lookup = {'BPTT': 'class=BPTrain approx=None f=Softmax(beta=0.1, train=True) normalize_weight=True',
#               'RBP': 'class=FPTrain approx=False f=Softmax(beta=0.1, train=True) normalize_weight=True',
#               'RBP-1': 'class=FPTrain approx=first f=Softmax(beta=0.1, train=True) normalize_weight=True',
#               'RBP-1H': 'class=FPTrain approx=first-heuristic f=Softmax(beta=0.1, train=True) normalize_weight=True'}
#%%
schemes = ['Memory', 'Small', 'Large']
algs = ['BPTT', 'RBP', 'RBP-1', 'RBP-1H']

def plot_loss(x_axis='iters', decimate=0):
    """x_axis = 'time' or 'iters"""
    schemes = scheme_lookup.keys()
    algs = alg_lookup.keys()
    fig, ax = plt.subplots(1,3)
    for i,scheme in enumerate(schemes):
        for j,alg in enumerate(algs):
            logdir = os.path.join(root, scheme_lookup[scheme], alg_lookup[alg])
            try:
                walltimes, iters, loss = get_data(logdir, 'train/loss')
                if scheme == 'Large' and alg == 'RBP':
                    walltimes = walltimes[:int(len(loss)/2)]
                    iters = iters[:int(len(loss)/2)]
                    loss = loss[:int(len(loss)/2)]
                reltimes = (walltimes-walltimes[0])/60
            except:
                print(f'{scheme}/{alg} not found')
                reltimes = iters = loss = []
            if x_axis == 'iters':
                x = iters
                xlabel = 'Iterations'
            elif x_axis == 'time':
                x = reltimes
                xlabel = 'Time (mins)'
            if decimate > 1:
                x = x[::decimate]
                loss = loss[::decimate]
            ax[i].semilogy(x, loss, label=alg, lw=1)
            ax[i].set_xlabel(xlabel)
            ax[i].set_title(scheme)
    ax[0].set_ylabel('Loss')
    ax[1].legend()
    fig.set_size_inches(6.5, 2)
    fig.tight_layout()

plot_loss('iters', decimate=5)
# plot_loss('time')

#%%
def plot_time():
    schemes = scheme_lookup.keys()
    algs = alg_lookup.keys()
    fig, ax = plt.subplots(1,3)
    for i,scheme in enumerate(schemes):
        for j,alg in enumerate(algs):
            logdir = os.path.join(root, scheme_lookup[scheme], alg_lookup[alg])
            try:
                walltimes, iters, loss = get_data(logdir, 'train/loss')
                if scheme == 'Large' and alg == 'RBP':
                    walltimes = walltimes[:int(len(loss)/2)]
                    iters = iters[:int(len(loss)/2)]
                    loss = loss[:int(len(loss)/2)]
                reltimes = (walltimes-walltimes[0])/60
            except:
                print(f'{scheme}/{alg} not found')
                reltimes = iters = loss = []
            ax[i].plot(iters, reltimes, label=alg)
            ax[i].set_xlabel('Iterations')
            ax[i].set_title(scheme)
    ax[0].set_ylabel('Time (min)')
    ax[0].legend()
    fig.set_size_inches(6.5, 2)
    fig.tight_layout()

plot_time()

#%%
schemes = scheme_lookup.keys()
algs = ['RBP', 'RBP-1', 'RBP-1H']

#TODO: get these from the actual config
data_values_config = cfg.Config({
    'class': 'MNISTDataset', #MNISTDataset, RandomDataset
    'include_test': False,
    'normalize' : 'data',
    'num_samples': 25,
    'balanced': False})
data_mode_config = cfg.Config({
    'classify': False,
    'perturb_entries': 0.5,
    'perturb_mode': 'rand',
    'perturb_value': 'min',
    })
train_data, test_data = data.get_aa_data(data_values_config, data_mode_config)
debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, select_classes=np.arange(1,10), n_per_class=1)

# debug_input = torch.rand(25,784,1)

fig, ax = plt.subplots(3,4)
plots._plot_rows(debug_input, ax=ax[0,0], cbar=False)
ax[0,0].set_title('Input')
[ax[i,0].axis('off') for i in [1,2]]
for i,scheme in enumerate(schemes):
    for j,alg in enumerate(algs):
        logdir = os.path.join(root, scheme_lookup[scheme], alg_lookup[alg])
        net = torch.load(os.path.join(logdir, 'net.pt'), map_location=torch.device('cpu'))
        net.fp_thres = 0.
        output = net((debug_input, ~debug_perturb_mask))
        plots._plot_rows(output, ax=ax[j,i+1], cbar=False)
        ax[0,i+1].set_title(scheme)
        ax[j,1].set_ylabel(alg)
fig.set_size_inches(6.5, 5)
fig.tight_layout()
