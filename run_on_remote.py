#!/usr/bin/env python3
import os, subprocess, argparse
import experiments

#Ginsburg cluster docs: https://confluence.columbia.edu/confluence/display/rcs/Ginsburg+-+Working+on+Ginsburg
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
REMOTE_SERVER = 'dt2586@ginsburg.rcs.columbia.edu' #must have ssh keys set up https://www.hostinger.com/tutorials/ssh/how-to-set-up-ssh-keys
REMOTE_SYNC_SERVER = 'dt2586@motion.rcs.columbia.edu'
REMOTE_PATH = '/burg/home/dt2586/assoc_mem' #'/burg/theory/users/dt2586/assoc_mem'

#%%select experiment
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment')
args = parser.parse_args()
if args.experiment is not None: #eg. running from CLI/batchscript
    expt = getattr(experiments, args.experiment)()
else: #default (eg. for running from IDE)
    expt = experiments.Stepsize_Onestep_Beta_Convergence()

#%%generate submit.sh file
time = expt.baseconfig.pop('slurm.time', '0-12:00:00') #days-hrs:mins:secs
cpus = expt.baseconfig.pop('slurm.cpus', 1)
# gpus = expt.baseconfig.pop('slurm.gpu', 0)
mem = expt.baseconfig.pop('slurm.mem', 16) #GB #TODO: make different per trial?

submit_sh = f"""#!/bin/sh

#SBATCH --account=theory
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem}gb
#SBATCH --time={time}
#SBATCH --array=0-{len(expt)-1}
#SBATCH --job-name={expt.__class__.__name__}
#SBATCH --output=slurm/slurm_%x_%a_%A.out

[[ ! -d slurm ]] && mkdir slurm
echo Queueing experiment {expt.__class__.__name__}, trial $SLURM_ARRAY_TASK_ID
srun python main.py -e {expt.__class__.__name__} -t $SLURM_ARRAY_TASK_ID
"""
#TODO: '#SBATCH --gpus-per-task={gpus}' doesn't work https://confluence.columbia.edu/confluence/display/rcs/Ginsburg+-+Job+Examples#GinsburgJobExamples-GPU(CUDAC/C++)
#TODO: test this: '#SBATCH --signal=SIGUSR1@90'
#TODO: use --cpus-per-gpu for gpu?

with open('submit.sh', 'w') as f:
    f.write(submit_sh)

#%%sync local-->remote
rsync = f'rsync -vv {LOCAL_PATH}/*.py {LOCAL_PATH}/submit.sh {REMOTE_SYNC_SERVER}:{REMOTE_PATH}'
print(f'Syncing...\n {rsync}')
subprocess.check_call(rsync, shell=True)

#%%run submit.sh on remote
ssh_sbatch = f'ssh {REMOTE_SERVER} "cd {REMOTE_PATH}; sbatch submit.sh"'
print(f'Batching...\n {ssh_sbatch}')
subprocess.check_call(ssh_sbatch, shell=True)

#%%run tensorboard on remote, port forward, and open in browser
# ssh_tensorboard = '' #TODO
# print('Starting tensorboard...\n {ssh_tensorboard}')
# subprocess.check_call(ssh_tensorboard, shell=True)
