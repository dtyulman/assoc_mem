import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler #for making custom color cycles
import numpy as np
import torch

MNIST_VPIX = 28
MNIST_HPIX = 28
MNIST_CLASSES = 10

def plot_loss_acc(loss, acc, test_loss=None, test_acc=None, iters=None,
                  title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2,1, sharex=True)

    if iters is None:
        iters = list(range(len(loss)))

    ax[0].plot(iters, loss, label='Train')
    if test_loss is not None:
        ax[0].plot(iters, test_loss, label='Test')
        ax[0].legend()
    ax[0].set_ylabel('Loss')

    ax[1].plot(iters, acc)
    if test_acc is not None:
        ax[1].plot(iters, test_acc)
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Accuracy')

    if title is not None:
        ax[0].set_title(title)

    return ax


def plot_data_batch(inputs, targets):
    fig, axs = plt.subplots(1,2)

    inputs_list = rows_to_images(inputs.squeeze(), pad_nan=True)
    targets_list = rows_to_images(targets.squeeze(), pad_nan=True)
    data = [inputs_list, targets_list]
    titles = ['Inputs', 'Targets']

    for ax, data_list, title in zip(axs, data, titles):
        #data: [B,M,1] or [B,Mc,1]
        grid = images_to_grid(data_list, vpad=1, hpad=1)
        im = ax.imshow(grid)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.axis('off')
        ax.set_title(title)


def plot_weights_mnist(W, max_n_rows=1024, plot_class=False, v=None, add_cbar=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    try:
        W = W.cpu().detach().numpy()
    except:
        pass

    if W.shape[0] > max_n_rows:
        print(f'Plotting only the first {max_n_rows}/{W.shape[0]} entries')
        W = W[:max_n_rows]

    if plot_class:
        img_list = rows_to_images(W, vpix=MNIST_VPIX+1, hpix=MNIST_HPIX, pad_nan=True)
    else:
        img_list = rows_to_images(W, vpix=MNIST_VPIX, hpix=MNIST_HPIX, drop_last=MNIST_CLASSES)
    grid = images_to_grid(img_list, vpad=1, hpad=1)

    v = np.nanmax(np.abs(grid)) if not v else v
    im = ax.imshow(grid, cmap='RdBu_r', vmin=-v, vmax=v)
    if add_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
    ax.axis('off')

    return ax


def plot_hidden_max_argmax(state_debug_history, n_per_class, apply_nonlin=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2,1)

    key = 'f' if apply_nonlin else 'h'
    h = state_debug_history[-1][key].detach()
    assert len(h.shape)==3, 'Hidden activity must have three dimensions: [B,N,1]'
    max_h_value, max_h_idx = torch.max(h, dim=1)

    ax[0].plot(max_h_idx, ls='', marker='.')
    ax[0].set_ylabel('Most active unit')

    ax[1].plot(max_h_value, ls='', marker='.')
    ax[1].set_ylabel(f'Activity $({key})$')
    ax[1].set_xlabel(f'Digit ({n_per_class} samples each)')

    size_data = h.shape[0] #B
    n_classes = size_data//n_per_class
    for a in ax.flatten():
        xticks = np.linspace(0,size_data, n_classes+1)
        for x in xticks:
            a.axvline(x, color='k', lw=0.5)
        a.set_xticks(xticks+n_per_class/2)
        a.set_xticklabels(list(range(10))+[None])


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
    state_trajectory = get_trajectory_by_key(state_debug_history, 'state').numpy().squeeze(-1) #[T,B,M]
    update_magnitude = np.linalg.norm(np.diff(state_trajectory, axis=0), axis=2)
    steps = np.array(range(1, len(state_trajectory)))

    ax = _plot_dynamics(update_magnitude, steps,
                              n_per_class=n_per_class, num_steps_train=num_steps_train, ax=ax)
    ax.set_ylabel('Update magnitude $|| \Delta \\vec{v}_t ||$')
    return ax


def plot_state_dynamics(state_debug_history, num_steps_plotted=20, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    state_trajectory = get_trajectory_by_key(state_debug_history, 'state').numpy().squeeze(-1) #[T,B,M]

    T,B,M = state_trajectory.shape
    num_steps_plotted = min(num_steps_plotted, T)
    steps = np.linspace(0, T-1, num_steps_plotted, dtype=int)
    state_trajectory = state_trajectory[steps] #[num_steps_plotted,B,M]

    state_trajectory_images = []
    for b in range(B):
        individual_state_trajectory = state_trajectory[:,b].squeeze() #[num_steps_plotted,M]
        state_trajectory_images += rows_to_images(individual_state_trajectory,
                                                  MNIST_HPIX, MNIST_VPIX, MNIST_CLASSES)
    state_trajectory_images = images_to_grid(state_trajectory_images,
                                             rows=B, cols=num_steps_plotted, vpad=1)

    v = np.nanmax(np.abs(state_trajectory_images))
    im = ax.imshow(state_trajectory_images, cmap='RdBu_r', vmin=-v, vmax=v)
    fig.colorbar(im, label='State value')
    # ax.axis('off')
    xticks = np.arange(MNIST_HPIX//2, MNIST_HPIX*num_steps_plotted+MNIST_HPIX//2, MNIST_HPIX)
    ax.set_xticks(xticks)
    ax.set_xticklabels(steps)
    ax.set_xlabel('Time steps')

    yticks = np.arange(MNIST_VPIX//2, MNIST_VPIX*B+MNIST_VPIX//2, MNIST_VPIX)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set_ylabel('Input')

    fig.set_size_inches(max(0.45*num_steps_plotted, 3), 0.41*B)
    fig.tight_layout()



def plot_energy_dynamics(state_debug_history, net,
                         n_per_class=1, num_steps_train=None, ax=None):
    state_trajectory = get_trajectory_by_key(state_debug_history, 'state')
    input = get_trajectory_by_key(state_debug_history, 'I')
    energy = net.energy(state_trajectory, input).squeeze(-1)

    ax = _plot_dynamics(energy,
                              n_per_class=n_per_class, num_steps_train=num_steps_train, ax=ax)
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


def rows_to_images(M, vpix=None, hpix=None, drop_last=0, pad_nan=True):
    """Convert a matrix M with R rows to a list of R vpix-by-hpix matrices (e.g. images).
    Optionally discard the drop_last entries of each row before reshaping it."""
    if drop_last > 0:
        M = M[:,:-drop_last]
    vpix, hpix = length_to_rows_cols(M.shape[1], vpix, hpix) #default to square-ish

    if pad_nan:
        length = M.shape[1]
        padding = vpix*hpix - length
        M = np.pad(M, ((0,0),(0,padding)), constant_values=float('nan'))
    return [r.reshape(vpix,hpix) for r in M]


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
