#third-party
import torch
import joblib

#custom
import plots
import data
from networks import ModernHopfield
from training import SGDTrain, FPTrain

#for my machine only TODO: remove
import os
if os.path.abspath('.').find('dtyulman') > -1:
    os.chdir('/home/dtyulman/projects/assoc_mem')

#??? Train on one step --> probably won't extrapolate to many
#??? Try "full" autoassociative task
#??? Can we remove the "dead" units?
#??? Normalize each weight vector per hidden unit to 1 --> removes the single giant hidden unit?

ax=None
legend = []
for input_mode in ['init', 'cont', 'init+cont']:
    for loss_mode in ['full', 'class']:
        label = f'input={input_mode}_loss={loss_mode}'
        legend.append(label)

        #%% train
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device =', device)

        #get dataset
        train_data, test_data = data.get_aa_mnist_classification_data(include_test=True)

        #define network
        input_size = train_data[0][0].numel()
        hidden_size = 50
        dt = .05
        net = ModernHopfield(input_size, hidden_size, fp_mode='iter', input_mode=input_mode, tau=1, beta=100, dt=dt, num_steps=50./dt, fp_thres=1e-9)
        # net = ModernHopfield(input_size, hidden_size, fp_mode='iter', tau=1, beta=1, dt=1, num_steps=1, fp_thres=1e10)
        net.to(device)
        logger = None

        #%%train network
        batch_size = 50
        debug_dataset = True #use tiny dataset which can be fully memorized

        if debug_dataset:
            train_data.dataset.data = train_data.dataset.data[:50]
            train_data.dataset.targets = train_data.dataset.targets[:50]
            if batch_size < len(train_data):
                test_loader = torch.utils.data.DataLoader(train_data, batch_size=50)
            else:
                test_loader = None
        else:
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # trainer = SGDTrain(net, train_loader, test_loader, logger=logger, only_class_loss=only_class_loss, print_every=10)
        trainer = FPTrain(net, train_loader, test_loader, logger=logger, loss_mode=loss_mode, print_every=10, lr=0.01)

        logger = trainer(epochs=1000, label=label)

        joblib.dump(logger, f'log_{label}.pkl')
        torch.save(net, f'net_{label}.pt')

#%% plot
# ax=None
# ll = []
# for log_fname in filter(lambda s: s.endswith('pkl'), next(os.walk('.'))[2]):
#     logger = joblib.load(log_fname)
#     label = log_fname[4:-4]
#     ax = plots.plot_loss_acc(logger['train_loss'], logger['train_acc'],
#                   # logger['test_loss'], logger['test_acc'],
#                   iters=logger['iter'],
#                   title='ModernHopfield (N={}), toy, (B={})'.format(net.hidden_size, batch_size),
#                   ax=ax)
#     ll.append(label)
# ax[0].legend(ll)

# net.to('cpu')
# axW = plots.plot_weights_mnist(net.W)
# axW.set_title(label)

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
