import matplotlib.pyplot as plt
import numpy as np


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
    V_PX_PLT = V_PX_IMG = 28
    H_PX_PLT = H_PX_IMG = 28
    C_PX_IMG = 10

    if ax is None:
        fig, ax = plt.subplots()

    try:
        W = W.cpu().detach().numpy()
    except:
        pass

    if W.shape[0] > max_n_rows:
        print(f'Plotting only the first {max_n_rows}/{W.shape[0]} entries')
        W = W[:max_n_rows]

    if W.shape[1] == V_PX_IMG*H_PX_IMG + C_PX_IMG:
        W_cls = W[:, -C_PX_IMG:]
        W = W[:, :-C_PX_IMG]
    elif W.shape[1] ==  V_PX_IMG*H_PX_IMG:
        W_cls = None
    else:
        raise ValueError(f'Unexpected number of columns: shape={W.shape}')

    if plot_class_pixels:
        H_PX_PLT = H_PX_IMG+1

    r = int(np.ceil(np.sqrt(W.shape[0])))
    c = int(np.round(np.sqrt(W.shape[0])))
    W_plt = np.full((r*V_PX_PLT, c*H_PX_PLT), float('nan'))
    for i in range(r):
        for j in range(c):
            idx = i*r + j
            if idx >= W.shape[0]:
                break
            w = W[idx].reshape(V_PX_IMG, H_PX_IMG)
            if plot_class_pixels:
                wc = np.full((V_PX_PLT,1), float('nan'))
                wc[:C_PX_IMG,:] = W_cls[idx].reshape(-1,1)
                w = np.concatenate((w,wc), axis=1)
            W_plt[i*V_PX_PLT:(i+1)*V_PX_PLT, j*H_PX_PLT:(j+1)*H_PX_PLT] = w

    im = ax.imshow(W_plt, cmap='RdBu_r')
    fig.colorbar(im)

    return ax
