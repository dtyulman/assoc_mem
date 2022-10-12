#built-in
import os, argparse

#third-party
import pytorch_lightning as pl
from torch.utils.data import DataLoader

try: #for my machine only TODO: remove
    os.chdir('/Users/danil/My/School/Columbia/Research/assoc_mem')
    import warnings
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
except (FileNotFoundError) as e:
    print(e.args)

#custom
import data, networks, experiments
import configs as cfg

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment')
parser.add_argument('-t', '--trial', type=int)
args = parser.parse_args()

#select experiment
expt = experiments.Associative_CIFAR10_Debug() #default (eg. for running from IDE)
if args.experiment is not None:
    expt = getattr(experiments, args.experiment)() #eg. running from CLI/batchscript

#initialize directory for saving if doesn't exist
if expt.baseconfig['train.save_logs']:
    use_existing = args.trial is not None and args.trial >= 2
    saveroot = cfg.initialize_savedir(expt, use_existing=use_existing)

#select trial (ie. detaconfigs) to run
configs, labels = expt.configs, expt.labels
if args.trial is not None:
    configs, labels = expt[args.trial]

#%%
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
        #saves to ./results/YYYY-MM-DD/ExperimentName/var1=delta1_var2=delta2/
        logger = pl.loggers.TensorBoardLogger(save_dir='./results', name=saveroot, version=label)
        logger.experiment.add_text('config', str(config))
    timer = pl.callbacks.Timer()
    printer = networks.Printer()
    trainer = pl.Trainer(max_epochs = config['train.epochs'],
                         log_every_n_steps = config['train.log_every'],
                         logger = logger,
                         enable_progress_bar = False,
                         enable_checkpointing = False,
                         callbacks = [timer, printer],
                         accelerator='auto', devices=1, auto_select_gpus=True
                         **config['train.trainer_kwargs'])
    trainer.fit(net, train_loader)
    print(f'Time elapsed{f" ({label})" if label else ""}: {timer.time_elapsed():.1f} sec')
