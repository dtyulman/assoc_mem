import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import higher

import plots
import components as nc
from networks import LargeAssociativeMemory
from data import AssociativeMNIST, ExceptionsDataset

# Net
class Autoencoder(nn.Module):
    def __init__(self, visible_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(visible_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, visible_size)
        self.softmax = nc.Softmax(beta=0.1, train=True)

    def forward(self, input):
        hidden = self.softmax(self.encoder(input))
        output = self.decoder(hidden)
        return output, hidden

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def plot_weights(self, weights='weights'):
        grid_e  = plots.images_to_grid(plots.rows_to_images(self.encoder.weight.detach()), vpad=1, hpad=1)
        grid_d  = plots.images_to_grid(plots.rows_to_images(self.decoder.weight.t().detach()), vpad=1, hpad=1)
        fig, ax = plots.plot_matrices([grid_e, grid_d], ['encoder', 'decoder'], ax_rows=1, cbar='individual')
        return fig, ax


# Task
def make_task_loader(reg=0, exc=1, n_reg='all', n_exc=3, batch_size=100):
    assoc_mnist = AssociativeMNIST(num_samples=None,
                                   select_classes=[reg, exc],
                                   n_per_class=[n_reg, n_exc],
                                   perturb_mask='rand',
                                   perturb_value='min')
    train_data = ExceptionsDataset(assoc_mnist, exceptions=[exc])
    return DataLoader(train_data, batch_size, shuffle=True)


# Losses
class LearnedLossWeights(nn.Module):
    def __init__(self, input_size, hidden_size, beta_init=10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nc.Softmax(beta=beta_init, train=False, dim=0)

    def forward(self, input):
        """input.shape = [B,N]"""
        h = F.relu(self.linear1(input))
        weights = self.softmax(self.linear2(h))
        return weights.squeeze() #[B]


def manual_loss_weights(is_exception, exception_loss_scaling):
    weights = ~is_exception + float(exception_loss_scaling)*is_exception
    weights /= weights.sum()
    return weights


class ExceptionsMSELoss(nn.Module):
    def __init__(self, compute_weights_fn):
        super().__init__()
        self.compute_weights = compute_weights_fn

    def forward(self, output, target, is_exception, *args, **kwargs):
        loss_per_item = F.mse_loss(output, target, reduction='none').mean(dim=1)
        loss_reg = loss_per_item[~is_exception].sum().item()
        loss_exc = loss_per_item[is_exception].sum().item()

        weights = self.compute_weights(*args, **kwargs)
        loss = (loss_per_item*weights).sum()
        return loss, loss_reg, loss_exc, weights


#%%
# reg=0
# exc=1
# train_loader = make_task_loader(reg, exc, n_reg=1999, n_exc=1, batch_size=100)
# net = Autoencoder(visible_size=train_loader.dataset[0][0].shape[0], hidden_size=100)
# loss_fn = ExceptionsMSELoss(manual_loss_weights)

# n_epochs = 1
# log = defaultdict(list)

# n_tot = len(train_loader.dataset)
# n_exc = len(train_loader.dataset.exceptions_idx)
# n_reg = n_tot-n_exc

# exception_loss_scaling = 2000# n_reg/n_exc
# optim = torch.optim.Adam(net.parameters(), lr=0.001)
# for epoch in range(n_epochs):
#     epoch_loss = epoch_loss_exc = epoch_loss_reg = 0
#     batch_losses = []
#     for batch_idx, batch in enumerate(train_loader):
#         input, target, perturb_mask, label, is_exception = batch
#         output, hidden = net(input)

#         compute_weights_args = is_exception, exception_loss_scaling
#         loss, loss_reg, loss_exc, loss_weights = loss_fn(output, target, is_exception, *compute_weights_args)

#         epoch_loss += loss.item()
#         epoch_loss_reg += loss_reg
#         epoch_loss_exc += loss_exc

#         optim.zero_grad()
#         loss.backward()
#         optim.step()

#         batch_losses.append(loss.item())

#         if batch_idx == 0 and epoch % max((n_epochs//5),1) == 0:
#             #plot weights
#             fig, ax = net.plot_weights()
#             fig.suptitle(f'epoch={epoch}')
#             #plot sample batch output
#             dataset = train_loader.dataset
#             input_exc, target_exc, perturb_mask_exc = dataset[dataset.exceptions_idx][0:3]
#             with torch.no_grad():
#                 output_exc, hidden_exc = net(input_exc)
#             n_exc = len(dataset.exceptions_idx)
#             input[-n_exc:] = input_exc
#             target[-n_exc:] = target_exc
#             output[-n_exc:] = output_exc
#             fig, ax = dataset.dataset.plot_batch(**{'Input':input, 'Target':target, 'Output':output.detach()})
#             fig.suptitle(f'epoch={epoch}')

#     print(f'batch_losses={[round(b, 3) for b in batch_losses]}')

#     epoch_loss_reg = epoch_loss_reg/n_reg
#     epoch_loss_exc = epoch_loss_exc/n_exc
#     if epoch % 10 == 0:
#         log['loss'].append(epoch_loss)
#         log['loss_reg'].append(epoch_loss_reg)
#         log['loss_exc'].append(epoch_loss_exc)
#         print(f'epoch {epoch+1}, loss={epoch_loss:.4f}, loss_reg={epoch_loss_reg:.4f}, loss_exc={epoch_loss_exc:.4f}')

# fig, ax = plt.subplots(3,1, sharex=True)
# for a, label in enumerate(['loss', 'loss_reg', 'loss_exc']):
#     ax[a].plot(log[label])
#     ax[a].set_ylabel(label)

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--n_tasks', type=int, default=5)
parser.add_argument('--ent_reg', type=int, default=1)
args = parser.parse_args()
print(args)

# Inner loop setup
n_epochs = args.n_epochs
n_tasks = args.n_tasks
task_specs = list(zip(range(0,n_tasks), range(1,n_tasks+1)))
dataloaders = [make_task_loader(reg, exc, n_reg=5797, n_exc=3, batch_size=100) for reg,exc in task_specs]
visible_size = dataloaders[0].dataset[0][0].shape[0]
hidden_size = 100
nets, optimizers = [], []
for task_i in range(n_tasks):
    nets.append( Autoencoder(visible_size=visible_size, hidden_size=hidden_size) )
    # nets.append(
    #     LargeAssociativeMemory(visible_size=visible_size, hidden_size=hidden_size,
    #                            visible_nonlin=nc.Spherical(), hidden_nonlin=nc.Softmax(),
    #                            )
    #             )
    optimizers.append( torch.optim.Adam(nets[task_i].parameters(), lr=0.001) )

learned_loss_weights = LearnedLossWeights(hidden_size, 50)
learned_loss_fn = ExceptionsMSELoss(learned_loss_weights)

# Outer loop setup
n_meta_iters = 100000
meta_optimizer = torch.optim.Adam(learned_loss_weights.parameters(), lr=0.001)
meta_loss_fn = ExceptionsMSELoss(manual_loss_weights)
entropy_regularizer = args.ent_reg #TODO: may want drop_last=True in dataloader to ensure equal batch size

#%% Run
for meta_iter in range(n_meta_iters):
    [net.reset_parameters() for net in nets]
    meta_optimizer.zero_grad()
    for epoch in range(n_epochs):
        # print(f' epoch {epoch}')
        avg_meta_loss = 0
        avg_batch_losses = 0
        for task_idx, (net, optimizer, dl, spec) in enumerate(zip(nets, optimizers, dataloaders, task_specs)):
            # print(f'  task {task_idx}')
            with higher.innerloop_ctx(net, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                batch_losses = []
                batch_has_exception = []
                # exception_loss_weights = []
                loss_weights_entropy = 0
                for batch_idx, batch in enumerate(dl):
                    # print(f'   batch {batch_idx}')
                    input, target, perturb_mask, label, is_exception = batch
                    visible, hidden = fnet(input)
                    loss, loss_reg, loss_exc, loss_weights = learned_loss_fn(visible, target, is_exception, hidden)
                    # loss, loss_reg, loss_exc, loss_weights = meta_loss_fn(visible, target, is_exception, is_exception, 2000)
                    loss_weights_entropy -= (loss_weights*torch.log(loss_weights)).sum()
                    if meta_iter % 100 == 0:
                        print(f'loss_weights(b={batch_idx})={loss_weights}')
                    diffopt.step(loss)
                    batch_losses.append(loss.item())
                    batch_has_exception.append(is_exception.any())
                    # if is_exception.any():
                    #     exception_loss_weights.append([round(w.item(),3) for w in loss_weights.unique()])
                avg_batch_losses += torch.tensor(batch_losses)

                loss_weights_entropy /= (batch_idx+1)
                meta_loss = 0
                for batch_idx, batch in enumerate(dl):
                    # print(f'   batch {batch_idx}')
                    input, target, perturb_mask, label, is_exception = batch
                    visible, hidden = fnet(input)
                    exc_scaling = len(dl.dataset)/len(dl.dataset.exceptions_idx)
                    _meta_loss, meta_loss_reg, meta_loss_exc, meta_loss_weights = \
                        meta_loss_fn(visible, target, is_exception, is_exception, exc_scaling)
                    meta_loss += _meta_loss
                avg_meta_loss += meta_loss.item()
                meta_loss += entropy_regularizer*loss_weights_entropy
                meta_loss.backward() #accumulates grads over epochs and tasks

        avg_meta_loss /= n_tasks
        avg_batch_losses /= n_tasks
        if avg_meta_loss == float('nan'):
            break
    meta_optimizer.step()

    if meta_iter % 50 == 0:
        avg_batch_losses = [round(b.item(),3) for b in avg_batch_losses]
        avg_batch_losses = [str(b) if bhe else b for bhe,b in zip(batch_has_exception, avg_batch_losses)]
        print(f'meta_iter={meta_iter+1}, meta_loss={avg_meta_loss:.4f}+{entropy_regularizer*loss_weights_entropy:.4f}, loss_weights_entropy={loss_weights_entropy:.4f}')
        print(f'batch_losses={avg_batch_losses}')
        # print(f'loss_weights={exception_loss_weights}')
