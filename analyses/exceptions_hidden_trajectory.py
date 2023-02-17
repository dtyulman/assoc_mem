import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import networks
from configs import Config
import data

root = '/Users/danil/My/School/Columbia/Research/assoc_mem/results/cluster_mountpoint/2022-11-28/AssociativeMNIST_Exceptions_0001'

data_config = Config({
        'class': data.AssociativeMNIST,
        'perturb_entries': 0.5,
        'perturb_mask': 'rand',
        'perturb_value': 'min',
        'include_test': False,
        'num_samples': None,
        'select_classes': [0, 1],
        'n_per_class': ['all', 3],
        'crop': False,
        'downsample': False,
        'normalize': True,
        })
train_data, test_data = data.get_data(**data_config)
train_loader = DataLoader(train_data, batch_size=4)

is_exception = [l.item() in net.exceptions for l in train_data.labels]
exceptions_idx = [idx for idx,is_ex in enumerate(is_exception) if is_ex]
input_ex, target_ex, perturb_mask_ex, label_ex = train_data[exceptions_idx]
batch = (input_ex, target_ex, perturb_mask_ex, label_ex)
batch_type = 'EXC_ONLY '

# input, target, perturb_mask, label = next(iter(train_loader))
# assert (label==0).all()
# batch = (input, target, perturb_mask, label)
# batch_type = 'REG_ONLY '

for beta in [1, 10]:
    for rg in [False]: #'unweighted_loss',
        for els in [100]: #10, 2000
            cfg_label = f'beta={beta} train_beta=False rescale_grads={rg} exception_loss_scaling={els}'

            path = os.path.join(root, cfg_label, 'checkpoints')
            ckpt = next(os.walk(path))[-1][0]
            path = os.path.join(path, ckpt)

            net = networks.ExceptionsMHN.load_from_checkpoint(path)
            fig, ax = net.plot_weights()
            fig.savefig(cfg_label + ' WEIGHTS.png', pad_inches=0, transparent=True)

            fig, ax = train_data.plot_batch(inputs=batch[0], targets=batch[1], outputs=net(batch[0], clamp_mask=~batch[2])[0].detach())
            fig.savefig(batch_type + cfg_label + ' BATCH.png', pad_inches=0, transparent=True)

            fig, ax = net.plot_hidden_trajectory(batch)
            fig.savefig(batch_type + cfg_label + '.png', pad_inches=0, transparent=True)
