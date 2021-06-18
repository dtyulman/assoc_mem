import matplotlib.pyplot as plt
from cycler import cycler #for making custom color cycles
import numpy as np
import torch


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


def plot_weights_mnist(W, max_n_rows=1024, plot_class_pixels=True, ax=None):
    MNIST_VPIX = 28
    MNIST_HPIX = 28
    MNIST_CLASSES = 10

    if ax is None:
        fig, ax = plt.subplots()

    try:
        W = W.cpu().detach().numpy()
    except:
        pass

    if W.shape[0] > max_n_rows:
        print(f'Plotting only the first {max_n_rows}/{W.shape[0]} entries')
        W = W[:max_n_rows]

    img_list = rows_to_images(W, vpix=MNIST_VPIX, hpix=MNIST_HPIX, drop_last=MNIST_CLASSES)
    grid = images_to_grid(img_list, vpad=1, hpad=1)

    v = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap='RdBu_r', vmin=-v, vmax=v)
    fig.colorbar(im)
    ax.axis('off')

    return ax


def plot_hidden_max_argmax(state_debug_history, n_per_class, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2,1)

    h = state_debug_history[-1]['h'].detach()
    assert len(h.shape)==3, 'Hidden activity must have three dimensions: [B,N,1]'
    max_h_value, max_h_idx = torch.max(h, dim=1)

    ax[0].plot(max_h_idx, ls='', marker='.')
    ax[0].set_ylabel('Most active unit')

    ax[1].plot(max_h_value, ls='', marker='.')
    ax[1].set_ylabel('Activity')
    ax[1].set_xlabel(f'Digit ({n_per_class} samples each)')

    size_data = h.shape[0] #B
    n_classes = size_data//n_per_class
    for a in ax.flatten():
        xticks = np.linspace(0,size_data, n_classes+1)
        for x in xticks:
            a.axvline(x, color='k', lw=0.5)
        a.set_xticks(xticks+n_per_class/2)
        a.set_xticklabels(list(range(10))+[None])


def plot_state_update_magnitude(state_debug_history, n_per_class, num_steps_train=None, ax=None):
    state_trajectory = np.array([state_debug['state'].detach().numpy()
                                 for state_debug in state_debug_history]).squeeze() #[T,B,M,1]
    update_magnitude = np.linalg.norm(np.diff(state_trajectory, axis=0), axis=2)
    steps = np.array(range(1, len(state_trajectory)))

    if ax is None:
        fig, ax = plt.subplots()
        size_data = state_trajectory.shape[1] #B
        n_classes = size_data//n_per_class
        cc = cycler(color=plt.cm.jet(np.linspace(0,1,n_classes)))
        ax.set_prop_cycle(cc)

    for i in range(n_per_class):
        ax.plot(steps, update_magnitude[:,i::n_per_class])
    ax.legend(list(range(10)))
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Update magnitude $|| \Delta \\vec{v}_t ||$')

    if num_steps_train:
        ax.axvline(num_steps_train, color='k', lw='0.5')


###############
### Helpers ###
###############
def rows_to_images(M, vpix=-1, hpix=-1, drop_last=0):
    """Convert a matrix <M> with R rows to a list of R <vpix>-by-<hpix> matrices (e.g. images).
    Optionally discard the <drop_last> entries of each row before reshaping it."""
    assert not(vpix==-1 and hpix==-1), 'Must specify at least one dimension of the image'
    if drop_last > 0:
        M = M[:,:-drop_last]
    return [r.reshape(vpix,hpix) for r in M]


def images_to_grid(img_list, rows=None, cols=None, vpad=0, hpad=0):
    """Convert a list of matrices (e.g. images) to a <rows>-by-<cols> grid of images (i.e. one
    large matrix). Optionally pad each matrix below/left with vpad/hpad rows/cols of NaNs"""
    n_imgs = len(img_list)
    if rows is None and cols is None:
        rows = int(np.ceil(np.sqrt(n_imgs)))
        cols = int(np.round(np.sqrt(n_imgs)))
    elif rows is None and cols is not None:
        rows = int(np.ceil(n_imgs/cols))
    elif rows is not None and cols is None:
        cols = int(np.ceil(n_imgs/rows))
    assert rows*cols >= n_imgs

    vpix = img_list[0].shape[0] + vpad
    hpix = img_list[0].shape[1] + hpad
    grid = np.full((rows*vpix, cols*hpix), fill_value=float('nan'))
    for i,img in enumerate(img_list):
        r = i//cols
        c = i%cols
        grid[r*vpix:(r+1)*vpix-vpad, c*hpix:(c+1)*hpix-hpad] = img

    return grid
