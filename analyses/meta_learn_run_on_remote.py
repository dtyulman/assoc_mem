#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:55:58 2023

@author: danil
"""
import os, subprocess
import experiments

#Ginsburg cluster docs: https://confluence.columbia.edu/confluence/display/rcs/Ginsburg+-+Working+on+Ginsburg
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
USERNAME = 'dt2586'
REMOTE_SERVER = 'dt2586@ginsburg.rcs.columbia.edu' #must have ssh keys set up https://www.hostinger.com/tutorials/ssh/how-to-set-up-ssh-keys
REMOTE_SYNC_SERVER = 'dt2586@motion.rcs.columbia.edu'
REMOTE_PATH = '/burg/theory/users/dt2586/assoc_mem' #'/burg/home/dt2586/assoc_mem' #

#%%select experiment and trials
experiment = experiments.AssociativeMNIST_Exceptions_Automatic()
# experiment_name = f'{experiment.__class__.__name__}'

for n_tasks in [1,5]:
    for n_epochs in [1,5]:
        for ent_reg in [0,1,10,100]:
            experiment_name = f'meta_learn_ntasks={n_tasks}_nepochs={n_epochs}_entreg={ent_reg}_trainB=False'

            #%%generate submit.sh file
            time = experiment.baseconfig.pop('slurm.time', '0-12:00:00') #days-hrs:mins:secs
            cpus = experiment.baseconfig.pop('slurm.cpus', 1)
            # gpus = experiment.baseconfig.pop('slurm.gpu', 0)
            mem = experiment.baseconfig.pop('slurm.mem', 16) #GB

            submit_sh = f"""#!/bin/sh

#SBATCH --account=theory
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem}gb
#SBATCH --time={time}
#SBATCH --job-name={experiment_name}
#SBATCH --output=slurm/slurm_%x_%a_%A.out

[[ ! -d slurm ]] && mkdir slurm
echo Queueing experiment {experiment_name}, trial $SLURM_ARRAY_TASK_ID
srun python -u meta_learn_exc_scaling.py --n_tasks {n_tasks} --n_epochs {n_epochs} --ent_reg {ent_reg}
"""

            with open('submit.sh', 'w') as f:
                f.write(submit_sh)

            #%%sync local-->remote
            rsync = f'rsync -vv {LOCAL_PATH}/*.py {LOCAL_PATH}/submit.sh {REMOTE_SYNC_SERVER}:{REMOTE_PATH}'
            print(f'Syncing...\n {rsync}')
            subprocess.check_call(rsync, shell=True)

            #%%run submit.sh on remote
            ssh_sbatch = f'ssh {REMOTE_SERVER} "cd {REMOTE_PATH}; sbatch submit.sh; squeue -u {USERNAME}"'
            print(f'Batching...\n {ssh_sbatch}')
            subprocess.check_call(ssh_sbatch, shell=True)


#%% Check log
# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=1_entreg=0_trainB=False_4294967294_5365035.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=1_entreg=100_trainB=False_4294967294_5365038.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=1_entreg=10_trainB=False_4294967294_5365037.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=1_entreg=1_trainB=False_4294967294_5365036.out'

# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=5_entreg=0_trainB=False_4294967294_5365039.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=5_entreg=100_trainB=False_4294967294_5365042.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=5_entreg=10_trainB=False_4294967294_5365041.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=1_nepochs=5_entreg=1_trainB=False_4294967294_5365040.out'

# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=1_entreg=0_trainB=False_4294967294_5365043.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=1_entreg=100_trainB=False_4294967294_5365046.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=1_entreg=10_trainB=False_4294967294_5365045.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=1_entreg=1_trainB=False_4294967294_5365044.out'

# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=5_entreg=0_trainB=False_4294967294_5365047.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=5_entreg=100_trainB=False_4294967294_5365050.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=5_entreg=10_trainB=False_4294967294_5365049.out'
# LOG_FILE = 'slurm_meta_learn_ntasks=5_nepochs=5_entreg=1_trainB=False_4294967294_5365048.out'


# LOG_FILE = f'{REMOTE_PATH}/slurm/{LOG_FILE}'
# subprocess.check_call(f'ssh {REMOTE_SERVER} "head {LOG_FILE}; echo; tail {LOG_FILE}"', shell=True)
# subprocess.check_call(f'ssh {REMOTE_SERVER} "cat {LOG_FILE}"', shell=True)

#%% Check queue
# subprocess.check_call(f'ssh {REMOTE_SERVER} "squeue -u {USERNAME}"', shell=True)

# %% Cancel all
# subprocess.check_call(f'ssh {REMOTE_SERVER} "scancel -u {USERNAME}"', shell=True)

#%%run tensorboard on remote, port forward, and open in browser
# ssh_tensorboard = '' #TODO
# print('Starting tensorboard...\n {ssh_tensorboard}')
# subprocess.check_call(ssh_tensorboard, shell=True)

#%%mount and launch tensorboard
# LOCAL_MOUNT_POINT = f'{LOCAL_PATH}/results/cluster_mountpoint'
# mount = f'sshfs {REMOTE_SYNC_SERVER}:{REMOTE_PATH}/results {LOCAL_MOUNT_POINT}'
# tensorboard = f'tensorboard --logdir {LOCAL_MOUNT_POINT}'
# subprocess.check_call(f'{mount}; {tensorboard}', shell=True)

#sshfs dt2586@motion.rcs.columbia.edu:/burg/home/dt2586/assoc_mem/results /Users/danil/My/School/Columbia/Research/assoc_mem/results/cluster_mountpoint
#sshfs dt2586@motion.rcs.columbia.edu:/burg/theory/users/dt2586/assoc_mem/results /Users/danil/My/School/Columbia/Research/assoc_mem/results/cluster_mountpoint

#tensorboard --logdir /Users/danil/My/School/Columbia/Research/assoc_mem/results/cluster_mountpoint
