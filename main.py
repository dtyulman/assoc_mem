#built-in
import os, argparse
import time

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
import callbacks as cb
import configs as cfg

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', help='One of the classes from experiments.py')
parser.add_argument('-t', '--trial', type=int, help='Entry in the flattened experiment loop')
args = parser.parse_args()#'-e Stepsize_Onestep_Beta_Convergence -t 1'.split())

#select experiment
if args.experiment is not None: #eg. running from CLI/batchscript
    expt = getattr(experiments, args.experiment)()
    experiment_summary_str = f'Running experiment: {expt.__class__.__name__} (cli input)'
else: #default (eg. for running from IDE)
    expt = experiments.Stepsize_Onestep_Beta_Convergence()
    experiment_summary_str = f'Running experiment: {expt.__class__.__name__} (default)'

#initialize directory for saving if doesn't exist
if expt.baseconfig['train.save_logs']:
    use_existing = args.trial is not None and args.trial >= 1
    if use_existing:
        #hack for running on cluster to give time for the 0th trial's savedir to get initilized
        #since multiple trials launched in parallel, so that subsequent trials can find it
        #and 'use_existing'
        time.sleep(5)
    saveroot = cfg.initialize_savedir(expt, use_existing=use_existing)

#select trial (ie. detaconfigs) to run
if args.trial is not None:
    configs, labels = expt[args.trial:args.trial+1]
    experiment_summary_str += f', trial #{args.trial}'
else:
    configs, labels = expt.configs, expt.labels
    experiment_summary_str += ', looping over all trials'

print(experiment_summary_str)

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
    printer = cb.Printer()
    trainer_kwargs = config.pop('train.trainer_kwargs', {})
    trainer = pl.Trainer(max_epochs = config['train.epochs'],
                         log_every_n_steps = config['train.log_every'],
                         logger = logger,
                         enable_progress_bar = False,
                         enable_checkpointing = False,
                         callbacks = [timer, printer],
                         accelerator='auto', devices=1, auto_select_gpus=True,
                         **trainer_kwargs)
    trainer.fit(net, train_loader)
    print(f'Time elapsed{f" ({label})" if label else ""}: {timer.time_elapsed():.1f} sec')
