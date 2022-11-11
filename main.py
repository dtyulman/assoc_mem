import os, argparse, time

import pytorch_lightning as pl
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import data, networks, experiments
import callbacks as cb
import configs as cfg

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', help='One of the classes from experiments.py')
parser.add_argument('-t', '--trial', type=int, help='Entry in the flattened experiment loop (1-based indexing)')
args = parser.parse_args()
assert args.trial is None or args.trial >= 1

#select experiment
if args.experiment is not None: #eg. running from CLI/batchscript
    expt = getattr(experiments, args.experiment)()
else: #default (eg. for running from IDE)
    expt = experiments.AssociativeMNIST_Exceptions_TwoBeta()

#initialize directory for saving if doesn't exist
if expt.baseconfig['train.save_logs']:
    use_existing = args.trial is not None and args.trial >= 2 #1-based indexing
    if use_existing:
        #TODO: fix, this is a hack for running on cluster to give time for the 1st trial's savedir
        #to get initilized since multiple trials launched in parallel, so that subsequent trials
        #can find it and correctly 'use_existing'
        time.sleep(5)
    saveroot = cfg.initialize_savedir(expt, use_existing=use_existing)

#select trial (ie. detaconfigs) to run
if args.trial is not None:
    configs, labels = expt[args.trial-1:args.trial] #1-based indexing
else:
    configs, labels = expt.configs, expt.labels


print(f'Running experiment: {expt.__class__.__name__} '
      f'{"(cli input)" if args.experiment is not None else "(script default)"}'
      f'{f", trial #{args.trial}" if args.trial is not None else ", looping over all trials"}')

#%%
for config, label in zip(configs, labels):
    print(config)
#%%
    #data
    train_data, test_data = data.get_data(**config['data'])
    batch_size = config['train'].pop('batch_size')
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=(batch_size!=len(train_data)))
    test_loader = DataLoader(test_data, batch_size=batch_size) if test_data else None

    #network
    NetClass = config['net'].pop('class')
    config['net'].update(NetClass.infer_input_size(train_data))
    net = NetClass(**config['net'], sparse_log_factor=config['train.sparse_log_factor'])
    if isinstance(net, networks.ExceptionsMHN):
        net.set_exception_loss_scaling(train_data)

    #training
    logger = False
    if config['train.save_logs']:
        #saves to ./results/YYYY-MM-DD/ExperimentName/var1=delta1_var2=delta2/
        logger = pl.loggers.TensorBoardLogger(save_dir='./results', name=saveroot, version=label)
        logger.experiment.add_text('config', str(config))
    timer = pl.callbacks.Timer()
    printer = cb.Printer()
    trainer_kwargs = config.pop('train.trainer_kwargs', {})
    trainer = pl.Trainer(max_epochs = config['train.max_epochs'],
                         max_steps = config['train.max_iters'],
                         log_every_n_steps = config['train.log_every'],
                         logger = logger,
                         enable_progress_bar = False,
                         callbacks = [timer, printer],
                         accelerator='auto', devices=1, auto_select_gpus=True,
                         **trainer_kwargs)
#%%
    trainer.fit(net, train_loader)
    print(f'Time elapsed{f" ({label})" if label else ""}: {timer.time_elapsed():.1f} sec')
