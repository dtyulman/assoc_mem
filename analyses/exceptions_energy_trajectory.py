import os, math

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

import networks
import components as nc
import data
from configs import Config
from plots import prep_axes

root = '/Users/danil/My/School/Columbia/Research/assoc_mem/results/cluster_mountpoint/2022-11-28/AssociativeMNIST_Exceptions_0001'

data_config = Config({
        'class': data.AssociativeMNIST,
        'perturb_entries': 0.5,
        'perturb_mask': 'rand',
        'perturb_value': 'min',
        'include_test': False,
        'num_samples': None,
        'select_classes': [0, 1],
        'n_per_class': ['all', 3],
        'crop': False,
        'downsample': False,
        'normalize': True,
        })
train_data, test_data = data.get_data(**data_config)


def get_batch(train_data, mode='', size=100, shuffle=False):
    is_exception = [l.item() in net.exceptions for l in train_data.labels]
    exceptions_idx = [idx for idx,is_ex in enumerate(is_exception) if is_ex]
    batch_exc = train_data[exceptions_idx]

    train_loader = DataLoader(train_data, batch_size=size, shuffle=shuffle)
    batch = next(iter(train_loader))
    label = batch[3]
    if not (label==0).all(): #try again
        print(label)
        return get_batch(train_data, mode=mode, size=size, shuffle=shuffle)

    if mode == '' or mode == 'mix':
        num_exc = len(batch_exc[0])
        for item, item_exc in zip(batch, batch_exc):
            item[-num_exc:] = item_exc
    elif mode == 'exc':
        batch = batch_exc
    return batch

#%%
rg = False #False, 'unweighted_loss',
els = 100 #1, 10, 100, 2000
for beta in [10]: #1,10
    cfg_label = f'beta={beta} train_beta=False rescale_grads={rg} exception_loss_scaling={els}'
    path = os.path.join(root, cfg_label, 'checkpoints')
    ckpt = next(os.walk(path))[-1][0]
    path = os.path.join(path, ckpt)

    net = networks.ExceptionsMHN.load_from_checkpoint(path,
                                                      visible_nonlin=nc.Spherical(),
                                                      visible_size=784)

#%%
# net = networks.LargeAssociativeMemoryWithCurrents(visible_size=net.visible.shape[0],
#                                                   hidden_size=net.hidden.shape[0],
#                                                   visible_nonlin = net.visible.nonlin,
#                                                   hidden_nonlin = net.hidden.nonlin,
#                                                   input_mode = net.input_mode,
#                                                   tau = net.visible.tau,
#                                                   dt = net.dt,
#                                                   max_steps = net.max_steps,
#                                                   state_mode='rates',
#                                                   dynamics_mode='hidden_prev')
# net.fc._weight.data = _net.fc.weight.clone()
# net.max_steps = 100
# with torch.no_grad():
#     state_trajectory = net(input, clamp_mask=~perturb_mask)


#%%
# net = networks.LargeAssociativeMemory(
net = networks.LargeAssociativeMemoryWithCurrents(dynamics_mode = 'grad_v',
    visible_size = 784,
    hidden_size = 100,
    visible_nonlin = nc.Spherical(),
    hidden_nonlin = nc.Softmax(beta=1),
    input_mode = 'init',
    tau = 1,
    dt = 0.5,
    max_steps = 100000)

# net.fc._weight.data = _net.fc._weight.data.clone()
# net.fc.normalize_mode = False
# net.fc._weight.data = net.fc._weight.data*100

B = 10
N = net.visible.shape[0]
batch = [torch.rand(B,N)*10,
          torch.rand(B,N),
          torch.rand(B,N).round().bool(),
          torch.randint(10, (B,))]
input, target, perturb_mask, label = batch

#%%
def plot_energy(E, dE, ax, plot='dE', title=''):
    if plot == 'E' or plot=='both':
        ax[0].plot(E)
        # ax[0].set_ylabel('E')

    if plot == 'dE' or plot == 'both':
        ax[-1].plot(dE)
        ax[-1].axhline(0, color='k', lw=0.3)
        if (~dE.isnan()).any():
            dE = dE[~dE.isnan()]
            if dE.max() > 0:
                ax[-1].set_ylim(-3*dE.max(), 1.5*dE.max())
            else:
                ax[-1].set_ylim(3*dE.max(), -1.5*dE.max())
        ax[-1].set_yticks([ax[-1].get_ylim()[0]*0.9, 0, ax[-1].get_ylim()[1]*0.9])
        # ax[-1].set_ylabel('$\Delta E$')

    # [ax[i].set_xlabel('Time') for i in [2,5]]

    [l.set_linewidth(0.5) for a in ax for l in a.get_lines()]
    ax[0].set_title(title, fontsize=8)
    # ax[0].ticklabel_format(scilimits=[-2,2])

#%%
vis_nonlin_list = [nc.Spherical()]
hid_nonlin_list = [nc.Softmax()]#, nc.Softmax(), nc.Sigmoid(), nc.Identity(), nc.Polynomial(3)]

outputs = {}
fig_E, ax_E = plt.subplots(2,3)
for i,input_mode in enumerate(['clamp', 'init']):
    for j, dynamics_mode in enumerate(['norm_grad_v', 'grad_v']):
        print(input_mode, dynamics_mode)
        # fig_E, ax_E = plt.subplots(len(vis_nonlin_list),len(hid_nonlin_list), squeeze=False, sharex='col')
        # fig_dE, ax_dE = plt.subplots(len(vis_nonlin_list),len(hid_nonlin_list), squeeze=False, sharex='col')

        # for i, vis_nonlin in enumerate(vis_nonlin_list):
        #     for j, hid_nonlin in enumerate(hid_nonlin_list):
        net.dynamics_mode = dynamics_mode
        net.input_mode = input_mode
                # net.visible.nonlin = vis_nonlin
                # net.hidden.nonlin = hid_nonlin

        with torch.no_grad():
            state_trajectory = net(input, clamp_mask=~perturb_mask, return_mode='trajectory')
            print(len(state_trajectory))
            outputs[f'{dynamics_mode}_{input_mode}'] = state_trajectory[-1]
            E = net.get_energy_trajectory(batch, state_trajectory=state_trajectory) #, debug=True)
        #     # Ef, Eh, Es, E = energy_trajectory
            dE = torch.diff(E, dim=0)

                # # fig, ax = plt.subplots(3,2, sharex=True)
                # # _ax = ax.flatten()
                # # for i, label in enumerate(['$E_{feat}$', '$E_{hid}$', '$E_{syn}$', '$E_{tot}$']):
                # #     _ax[i].plot(energy_trajectory[i])
                # #     _ax[i].set_ylabel(label)

        title = f'{net.visible.nonlin}/{net.hidden.nonlin}, {net.input_mode}, dyn={net.dynamics_mode}'\
                        .replace('Polynomial(n=', 'Poly(')\
                        .replace('Identity', 'Id')\
                        .replace('Spherical', 'Sph')\
                        .replace('Softmax(beta', 'Sfmx($\\beta$')\
                        .replace('Sigmoid', 'Sig')
        plot_energy(E, dE, ax_E[i:i+1, j], plot='E', title=title)
        # plot_energy(E, dE, ax_dE[i:i+1, j], plot='dE')

#%% Does clamping matter?
    # batch = get_batch(train_data, size=9)
    # net.max_steps = 100
    # net.converged_thres = 0
    input, target, perturb_mask = batch[0:3]
    # input /= input.norm(dim=-1, keepdim=True)

    # generate clamped/unclamped trajectories
    with torch.no_grad():
        net.input_mode = 'clamp'
        state_trajectory = net(input, clamp_mask=~perturb_mask, return_mode='trajectory')
        E = net.get_energy_trajectory(batch, state_trajectory=state_trajectory)

        net.input_mode = 'init'
        state_trajectory_unclamped = net(input, return_mode='trajectory')
        E_unclamped = net.get_energy_trajectory(batch, state_trajectory=state_trajectory_unclamped)

#%%
    # plot state at 5 intermediate timepoints
    fig, ax = plt.subplots(2,6)
    ax = ax.flatten()
    t_list = torch.logspace(0, math.log10(len(state_trajectory)), 5).int()-1
    state_trajectory_plot = {f't={t}':state_trajectory[t][0] for t in t_list}
    state_trajectory_plot['Target'] = target
    train_data.plot_batch(**state_trajectory_plot, ax=ax[0:6])
    ax[0].set_ylabel('Clamped')

    t_list = torch.logspace(0, math.log10(len(state_trajectory_unclamped)), 5).int()-1
    state_trajectory_plot = {f't={t}':state_trajectory_unclamped[t][0] for t in t_list}
    state_trajectory_plot['Target'] = target
    train_data.plot_batch(**state_trajectory_plot, ax=ax[6:12])
    ax[6].set_ylabel('Unclamped')

    # plot energy
    fig, ax = plt.subplots(2,2, sharex=True)
    ax[0,0].plot(E)
    ax[0,0].set_ylabel('E (clamped)')
    ax[1,0].plot(E_unclamped)
    ax[1,0].set_ylabel('E (unclamped)')
    with torch.no_grad():
        ax[0,1].plot(torch.diff(E, dim=0))
        ax[0,1].axhline(0, color='k', lw=0.3)
        ax[0,1].set_ylabel('$\Delta E$ (clamped)')
        ax[0,1].set_ylim(-2e-6, 2e-6)
        ax[1,1].plot(torch.diff(E_unclamped, dim=0))
        ax[1,1].axhline(0, color='k', lw=0.3)
        ax[1,1].set_ylabel('$\Delta E$ (unclamped)')
        ax[1,1].set_ylim(-2e-6, 2e-6)
    for a in ax.flatten():
        [l.set_color('r') for l in a.get_lines()[0:6]]
        [l.set_color('b') for l in a.get_lines()[6:9]]
        [l.set_linewidth(0.5) for l in a.get_lines()]
