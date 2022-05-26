from copy import deepcopy
import math

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from networks import ConvolutionalMem


#data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torchvision.datasets.CIFAR10(root='./data/CIFAR10').data
data = torch.tensor(data).transpose(-2,-1).transpose(-3,-2)
data = (data-data.min())/(data.max()-data.min())
data = data.to(device)

perturb_frac = 0.1
perturbed_data = deepcopy(data)
perturbed_data[torch.rand_like(perturbed_data) <= perturb_frac] = 0
perturbed_data = perturbed_data.to(device)

batch_size = 128
n_batches = math.floor(len(data)/batch_size)

#network
x_size = 32
x_channels = 3
y_channels = 20
z_size = 100
kernel_size = 8
stride = 1
dt = 0.1
fp_thres = 0
max_num_steps = 500
net = ConvolutionalMem(x_size, x_channels,  y_channels=y_channels,
                       kernel_size=kernel_size, stride=stride, z_size=z_size,
                       dt=dt, fp_thres=fp_thres, max_num_steps=max_num_steps)
net.to(device)

#optimizer
optim = torch.optim.Adam(net.parameters())
loss_fn = torch.nn.MSELoss()

#training
epochs = 1000
print_every = 10
writer = SummaryWriter('./results/2022-05-20/min/000')
it = 0
#%%
for ep in range(epochs):
    for i in range(n_batches):
        it += 1
        input = perturbed_data[i*batch_size:(i+1)*batch_size]
        target = data[i*batch_size:(i+1)*batch_size]

        output = net((input, None))
        loss = loss_fn(output, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        writer.add_scalar('train/loss', loss.item())
        if it%print_every == 0:
            print(f'ep:{ep} it:{it}, loss={loss}')

        torch.cuda.empty_cache()
