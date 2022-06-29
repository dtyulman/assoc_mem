import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler #for making custom color cycles
import numpy as np
import torch
import torchvision
import math


def plot_loss_acc(logger, plot_test=True, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2,1, sharex=True)

    logger.plot_key('train/loss', label='Train', ax=ax[0])
    logger.plot_key('train/acc', label='Train', ax=ax[1])
    ax[0].set_ylabel('Loss')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Accuracy')
    if title is not None:
        ax[0].set_title(title)

    if plot_test:
        if 'test/loss' in logger:
            logger.plot_key('test/loss', label='Test', ax=ax[0])
            ax[0].legend()
        if 'test/acc' in logger:
            logger.plot_key('test/acc', label='Test', ax=ax[1])
    return ax


def _plot_rows(mat, drop_last=0, pad_nan=True, title='', ax=None,
               cbar=True, cmap='RdBu_r', vmin=None, vmax=None):
    fig, ax = prep_axes(ax)
    mat = mat.squeeze(dim=-1)

    if len(mat.shape) == 2: #[B,M] for data batch or [N,M] for a weight matrix
        try:
            mat = mat.cpu().detach().numpy()
        except:
            pass
        imgs = rows_to_images(mat, drop_last=drop_last, pad_nan=pad_nan)
        grid = images_to_grid(imgs, vpad=1, hpad=1)
    elif len(mat.shape) == 4: #[B,C,L,L] for image data batch
        assert drop_last==0
        cbar=False
        rows,_ = length_to_rows_cols(len(mat))
        grid = torchvision.utils.make_grid(mat, rows, normalize=True, padding=1,
                                           pad_value=torch.tensor(float('nan')))
        try:
            grid = grid.cpu().detach().numpy()
        except:
            pass
        grid = np.transpose(grid, (1,2,0))


    if vmin is None and vmax is None and cmap == 'RdBu_r':
        vmax = np.nanmax(np.abs(grid))
        vmin = -vmax
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)


    [ax.spines[spine].set_visible(False) for spine in ['top', 'bottom', 'left', 'right']]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    return ax


def plot_data_batch(inputs, targets, outputs=None, ax=None):
    if outputs is None:
        fig, axs = prep_axes(ax,1,2, sharex=True, sharey=True)
        v = max(inputs.abs().max(), targets.abs().max())
        _plot_rows(inputs,  ax=axs[0], title='Inputs',  cmap='RdBu_r', vmin=-v, vmax=v)
        _plot_rows(targets, ax=axs[1], title='Targets', cmap='RdBu_r', vmin=-v, vmax=v)
        scale_fig(fig, 1.5)
    else:
        fig, axs = prep_axes(ax,1,3, sharex=True, sharey=True)
        v = max(inputs.abs().max(), targets.abs().max(), outputs.abs().max())
        _plot_rows(inputs,  ax=axs[0], title='Inputs',  cmap='RdBu_r', vmin=-v, vmax=v)
        _plot_rows(targets, ax=axs[1], title='Targets', cmap='RdBu_r', vmin=-v, vmax=v)
        _plot_rows(outputs, ax=axs[2], title='Outputs', cmap='RdBu_r', vmin=-v, vmax=v)
        scale_fig(fig, 2)
    fig.tight_layout()
    return axs


def plot_fixed_points(net, num_fps=100, inputs=None, drop_last=0, ax=None):
    if inputs is None:
        hidden_size, input_size = net.W.shape #[N,M]
        inputs = torch.rand(num_fps, input_size, 1)

    num_steps = net.max_num_steps
    fp_thres = net.fp_thres
    input_mode = net.input_mode

    net.num_steps = 10000
    net.fp_thres = 1e-10
    net.input_mode = 'init'

    outputs = net(inputs)

    net.max_num_steps = num_steps
    net.fp_thres = fp_thres
    net.input_mode = input_mode

    return _plot_rows(outputs, drop_last=drop_last, title='Fixed points', ax=ax)



def plot_weights(net=None, _W=None, W=None, drop_last=0, ax=None):
    if net is not None:
        assert _W is None and W is None, 'Do not provide weight matrix if passing in net object'
        _W = net._W
        if net.normalize_weight:
            W = net.W

    if W is not None:
        fig, ax = prep_axes(ax,1,2, sharex=True, sharey=True)
        _plot_rows(_W, ax=ax[0], drop_last=drop_last, title='_W (raw)')
        _plot_rows(W, ax=ax[1], drop_last=drop_last, title='W (normalized)')
        scale_fig(fig, 1.5)
    else:
        fig, ax = prep_axes(ax)
        _plot_rows(_W, drop_last=drop_last, title='Raw (no normalization)', ax=ax)
    # fig.tight_layout()

    return ax


def plot_hidden_max_argmax(state_debug_history, n_per_class=None, apply_nonlin=True, ax=None):
    fig, ax = prep_axes(ax,2,1)

    key = 'f' if apply_nonlin else 'h'
    h = state_debug_history[-1][key].detach().to('cpu') #[B,N,1]
    assert len(h.shape)==3, 'Hidden activity must have three dimensions: [B,N,1]'
    max_h_value, max_h_idx = torch.max(h.squeeze(dim=-1), dim=1) #[B,N]-->([B],[B])

    if n_per_class is None:
        max_h_idx, sort = torch.sort(max_h_idx)
        max_h_value = max_h_value[sort]

    ax[0].plot(max_h_idx, ls='', marker='.')
    ax[0].set_ylabel('Most active unit')

    ax[1].plot(max_h_value, ls='', marker='.')
    ax[1].set_ylabel(f'Activity of max unit $({key})$')
    if n_per_class is None:
        ax[1].set_xlabel('Sample')
    else:
        ax[1].set_xlabel(f'Class ({n_per_class} samples each)')
        size_data = h.shape[0] #B
        n_classes = size_data//n_per_class
        for a in ax.flatten():
            xticks = np.linspace(0,size_data, n_classes+1)
            for x in xticks:
                a.axvline(x, color='k', lw=0.5)
            a.set_xticks(xticks+n_per_class/2)
            a.set_xticklabels(list(range(n_classes))+[None])

    return ax


def _plot_dynamics(var, steps=None, n_per_class=1, num_steps_train=None, legend='auto', ax=None):
    """Plots the evolution of <var> as dynamics progresses over time <steps>"""
    size_data = var.shape[1] #B
    n_classes = size_data//n_per_class

    if ax is None:
        fig, ax = plt.subplots()
        cc = cycler(color=plt.cm.jet(np.linspace(0,1,n_classes)))
        ax.set_prop_cycle(cc)

    if steps is None:
        steps = np.array(range(len(var)))

    for i in range(n_per_class):
        ax.plot(steps, var[:,i::n_per_class])
    if legend == 'auto':
        legend = list(range(n_classes))
    ax.legend(legend)
    ax.set_xlabel('Time steps')

    if num_steps_train and steps[-1]>num_steps_train:
        ax.axvline(num_steps_train, color='k', lw='0.5')
    return ax


def plot_state_update_magnitude_dynamics(state_debug_history,
                                         n_per_class=1, num_steps_train=None, ax=None):
    state_trajectory = get_trajectory_by_key(state_debug_history, 'v').numpy().squeeze(-1) #[T,B,M]
    update_magnitude = np.linalg.norm(np.diff(state_trajectory, axis=0), axis=2)
    steps = np.array(range(1, len(state_trajectory)))

    ax = _plot_dynamics(update_magnitude, steps,
                              n_per_class=n_per_class, num_steps_train=num_steps_train, ax=ax)
    ax.set_ylabel('Update magnitude $|| \Delta \\vec{v}_t ||$')
    return ax


def plot_state_dynamics(state_debug_history, num_steps_plotted=20, targets=None, ax=None):
    fig, ax = prep_axes(ax)

    state_trajectory = get_trajectory_by_key(state_debug_history, 'v').numpy().squeeze(-1) #[T,B,M]
    if targets is not None: #plot error instead of state
        title = 'Error (v-t)'
        state_trajectory = state_trajectory - targets.squeeze(-1).unsqueeze(0).numpy() #[T,B,M]-[B,M]
    else:
        title = 'State (v)'

    T,B,M = state_trajectory.shape
    hpix = vpix = int(math.sqrt(M))

    num_steps_plotted = min(num_steps_plotted, T)
    steps = np.linspace(0, T-1, num_steps_plotted, dtype=int)
    state_trajectory = state_trajectory[steps] #[T,B,M]->[num_steps_plotted,B,M]

    state_trajectory_images = []
    for b in range(B):
        individual_state_trajectory = state_trajectory[:,b] #[num_steps_plotted,M]
        state_trajectory_images += rows_to_images(individual_state_trajectory, #vpix=vpix,
                                                  hpix=hpix)#, drop_last=MNIST_CLASSES)
    state_trajectory_images = images_to_grid(state_trajectory_images,
                                             rows=B, cols=num_steps_plotted, vpad=1)

    v = np.nanmax(np.abs(state_trajectory_images))
    im = ax.imshow(state_trajectory_images, cmap='RdBu_r', vmin=-v, vmax=v)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax)

    xticks = np.arange(hpix//2, hpix*num_steps_plotted+hpix//2, hpix)
    ax.set_xticks(xticks)
    ax.set_xticklabels(steps)
    ax.set_xlabel('Time steps')

    yticks = np.arange(vpix//2, vpix*B+vpix//2, vpix)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set_ylabel('Input')

    ax.set_title(title)

    fig.set_size_inches(max(0.45*num_steps_plotted, 3), 0.41*B)
    fig.tight_layout()


def plot_energy_dynamics(state_debug_history, net,
                         n_per_class=1, num_steps_train=None, ax=None):
    state_trajectory = get_trajectory_by_key(state_debug_history, 'v') #[T,B,M]
    input = get_trajectory_by_key(state_debug_history, 'I')
    energy = net.energy(state_trajectory, input).squeeze(-1)

    ax = _plot_dynamics(energy, n_per_class=n_per_class, num_steps_train=num_steps_train, ax=ax)
    ax.set_ylabel('Energy')

    if (energy>0).any() and (energy<0).any():
        ax.axhline(0, color='k', lw='0.5', ls='--')
    return ax


def plot_hidden_dynamics(state_debug_history, apply_nonlin=True, transformation='max',
                         n_per_class=1, num_steps_train=None, ax=None):
    hidden_trajectory = get_trajectory_by_key(state_debug_history, 'f' if apply_nonlin else 'h').squeeze(-1) #[T,B,N]

    if transformation == 'max':
        transformed_trajectory, _ = torch.max(hidden_trajectory, dim=-1) #[T,B]
    elif transformation == 'mean':
        transformed_trajectory = hidden_trajectory.mean(dim=-1) #[T,B]
    else:
        raise ValueError(f"Invalid transformation: '{transformation}'")

    ax = _plot_dynamics(transformed_trajectory,
                        n_per_class=n_per_class, num_steps_train=num_steps_train, ax=ax)

    if transformation == 'max':
        N = hidden_trajectory.shape[-1]
        ax.axhline(1./N, color='k', lw='0.5', ls='--')
    elif transformation == 'mean':
        std = hidden_trajectory.std(dim=-1) #[T,B]
        top = transformed_trajectory+std
        bot = transformed_trajectory-std
        lines = ax.get_lines()
        for i, (y1, y2) in enumerate(zip(top.T, bot.T)):
            line = lines[i]
            x = line.get_xdata()
            ax.fill_between(x, y1, y2, alpha=0.3, color=line.get_color())

    ax.set_ylabel('{}($\\vec{{f}}$)'.format(transformation))
    return ax


###############
### Helpers ###
###############
def get_trajectory_by_key(state_debug_history, key):
    trajectory = torch.stack([state_debug[key].detach()
                            for state_debug in state_debug_history])
    return trajectory #[T,B,*] e.g. [T,B,M,1] or [T,B,N,1]


def rows_to_images(mat, vpix=None, hpix=None, drop_last=0, pad_nan=True):
    """Convert a matrix mat with R rows to a list of R vpix-by-hpix matrices (e.g. images).
    Optionally discard the drop_last entries of each row before reshaping it."""
    if drop_last > 0:
        mat = mat[:,:-drop_last]
    length = mat.shape[1]
    vpix, hpix = length_to_rows_cols(length, vpix, hpix) #default to square-ish

    if pad_nan:
        padding = vpix*hpix - length
        mat = np.pad(mat, ((0,0),(0,padding)), constant_values=float('nan'))
    return [r.reshape(vpix,hpix) for r in mat]


def images_to_grid(img_list, rows=None, cols=None, vpad=0, hpad=0):
    """Convert a list of matrices (e.g. images) to a <rows>-by-<cols> grid of images (i.e. one
    large matrix). Optionally pad each matrix below/left with vpad/hpad rows/cols of NaNs"""
    vpix = img_list[0].shape[0] + vpad
    hpix = img_list[0].shape[1] + hpad
    rows, cols = length_to_rows_cols(len(img_list), rows, cols)
    grid = np.full((rows*vpix, cols*hpix), fill_value=float('nan'))
    for i,img in enumerate(img_list):
        r = i//cols
        c = i%cols
        grid[r*vpix:(r+1)*vpix-vpad, c*hpix:(c+1)*hpix-hpad] = img
    return grid


def length_to_rows_cols(length, rows=None, cols=None):
    if rows is None and cols is None:
        rows = int(np.ceil(np.sqrt(length)))
        cols = int(np.round(np.sqrt(length)))
    elif rows is None and cols is not None:
        rows = int(np.ceil(length/cols))
    elif rows is not None and cols is None:
        cols = int(np.ceil(length/rows))
    assert rows*cols >= length #sanity
    return rows, cols


def prep_axes(ax=None, nrows=1, ncols=1, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, **kwargs)
    else:
        if ncols>1 or nrows>1: #sanity check if specifying nrows or ncols
            assert len(ax.flatten()) == nrows*ncols
        fig = ax.get_figure()
    return fig, ax


def scale_fig(fig, wscale=1, hscale=1):
    w,h = fig.get_size_inches()
    fig.set_size_inches(wscale*w, hscale*h)
    return fig
