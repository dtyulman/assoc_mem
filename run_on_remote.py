import os, subprocess
import experiments

#Ginsburg cluster docs: https://confluence.columbia.edu/confluence/display/rcs/Ginsburg+-+Working+on+Ginsburg
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
USERNAME = 'dt2586'
REMOTE_SERVER = 'dt2586@ginsburg.rcs.columbia.edu' #must have ssh keys set up https://www.hostinger.com/tutorials/ssh/how-to-set-up-ssh-keys
REMOTE_SYNC_SERVER = 'dt2586@motion.rcs.columbia.edu'
REMOTE_PATH = '/burg/theory/users/dt2586/assoc_mem' #'/burg/home/dt2586/assoc_mem' #

#%%select experiment and trials
experiment = experiments.AssociativeMNIST_Exceptions_TwoBeta()
trials = f'1-{len(experiment)}' #default, all trials

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
#SBATCH --array={trials}
#SBATCH --job-name={experiment.__class__.__name__}
#SBATCH --output=slurm/slurm_%x_%a_%A.out

[[ ! -d slurm ]] && mkdir slurm
echo Queueing experiment {experiment.__class__.__name__}, trial $SLURM_ARRAY_TASK_ID
srun python -u main.py -e {experiment.__class__.__name__} -t $SLURM_ARRAY_TASK_ID
"""
#TODO: test this: '#SBATCH --signal=SIGUSR1@90'
#TODO: '#SBATCH --gpus-per-task={gpus}' doesn't work https://confluence.columbia.edu/confluence/display/rcs/Ginsburg+-+Job+Examples#GinsburgJobExamples-GPU(CUDAC/C++)
#TODO: use --cpus-per-gpu for gpu?

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

#%%
# subprocess.check_call(f'ssh {REMOTE_SERVER} "squeue -u {USERNAME}"', shell=True)


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
