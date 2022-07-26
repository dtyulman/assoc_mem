#built-in
import os, argparse

#third-party
import pytorch_lightning as pl
from torch.utils.data import DataLoader

#for my machine only TODO: remove
try:
    os.chdir('/Users/danil/My/School/Columbia/Research/assoc_mem')
except (FileNotFoundError) as e:
    print(e.args)

#custom
import data, networks, experiments
import configs as cfg

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment')
args = parser.parse_args()

CurrentExperiment = experiments.Associative_CIFAR10_Debug()
if args.experiment is not None:
    CurrentExperiment = getattr(experiments, args.experiment)

#%%
saveroot = None
if CurrentExperiment.baseconfig['train.save_logs']:
    saveroot = cfg.initialize_savedir(CurrentExperiment.baseconfig)
configs, labels = cfg.flatten_config_loop(CurrentExperiment.baseconfig,
                                          CurrentExperiment.deltaconfigs,
                                          CurrentExperiment.mode)

for config, label in zip(configs, labels):
    print(config)

    #data
    train_data, test_data = data.get_data(**config['data'])
    batch_size = config['train'].pop('batch_size')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size) if test_data else None

    #network
    NetClass = getattr(networks, config['net'].pop('class'))
    config['net'].update(NetClass.infer_input_size(train_data))
    net = NetClass(**config['net'], sparse_log_factor=config['train.sparse_log_factor'])

    #training
    logger = False
    if config['train.save_logs']:
        logger = pl.loggers.TensorBoardLogger('./results', name=saveroot, version=label)
        logger.experiment.add_text('config', str(config))
    timer = pl.callbacks.Timer()
    printer = networks.Printer()
    trainer = pl.Trainer(max_epochs = config['train.epochs'],
                         log_every_n_steps = config['train.log_every'],
                         logger = logger,
                         enable_progress_bar = False,
                         enable_checkpointing = False,
                         callbacks = [timer, printer],
                         accelerator='auto', devices=1, auto_select_gpus=True)
    trainer.fit(net, train_loader)
    print(f'Time elapsed{f" ({label})" if label else ""}: {timer.time_elapsed():.1f} sec')

#%%
inputs, targets, perturb_mask = next(iter(train_loader))
train_data.plot_batch(inputs, targets, outputs=net(inputs)[0])
#%%
net.plot_weights()
# %% plot
# plots.plot_loss_acc(logger)

# #%%
# # plots.plot_weights(net, drop_last=0)

# #%%
# n_per_class = 10
# debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)

# try:
#     state_debug_history = net((debug_input, ~debug_perturb_mask), debug=True)
#     debug_output = state_debug_history[-1]['v']
#     plots.plot_hidden_max_argmax(state_debug_history, n_per_class, apply_nonlin=True)
# except:
#     debug_output = net((debug_input, ~debug_perturb_mask))

# plots.plot_data_batch(debug_input, debug_target, debug_output)

# #%%

# plots.plot_data_batch(debug_input, debug_target)
# #%%
# # n_per_class = 1
# # debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)

# # max_num_steps = net.max_num_steps
# # # fp_thres = net.fp_thres
# # # check_converged = net.check_converged
# # # net.max_num_steps = 1e10#int(100/net.dt)
# # # net.fp_thres = 1e-9
# # # net.check_converged = True

# # state_debug_history = net((debug_input, ~debug_perturb_mask), debug=True)

# # # net.max_num_steps = max_num_steps
# # # net.fp_thres = fp_thres
# # # net.check_converged = check_converged

# # fig, ax = plt.subplots(2,2, sharex=True)
# # ax = ax.flatten()

# # plots.plot_state_update_magnitude_dynamics(state_debug_history, n_per_class, max_num_steps, ax=ax[0])
# # plots.plot_energy_dynamics(state_debug_history, net, num_steps_train=max_num_steps, ax=ax[1])
# # plots.plot_hidden_dynamics(state_debug_history, transformation='max', num_steps_train=max_num_steps, ax=ax[2])
# # plots.plot_hidden_dynamics(state_debug_history, apply_nonlin=False, transformation='mean', ax=ax[3])

# # [a.set_xlabel('') for a in ax[0:2]]
# # [a.legend_.remove() for a in ax[0:-1]]
# # plots.scale_fig(fig, 1.5, 1.5)
# # fig.tight_layout()

# #%%
# fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
# plots.plot_state_dynamics(state_debug_history, ax=ax[0])
# plots.plot_state_dynamics(state_debug_history, targets=debug_target, ax=ax[1]) #plot error instead of state
# plots.scale_fig(fig, 1.6, 3.5)
# fig.tight_layout()

# #%%
# debug_input, debug_target, debug_perturb_mask = data.get_aa_debug_batch(train_data, n_per_class=n_per_class)
# fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
# for i,beta in enumerate([5, 5.5, 10]):
#     beta_orig = net.f.beta.item()

#     net.f.beta.data = torch.tensor(beta)
#     plots.plot_fixed_points(net, num_fps=100, drop_last=0, ax=ax[0,i])
#     ax[0,i].set_title(f'beta={beta}')
#     plots.plot_fixed_points(net, inputs=debug_input, drop_last=0, ax=ax[1,i])
#     ax[1,i].set_title('')

#     net.f.beta.data = torch.tensor(beta_orig)
# ax[0,0].set_ylabel('Rand init')
# ax[1,0].set_ylabel('Data init')
