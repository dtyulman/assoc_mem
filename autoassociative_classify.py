from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from utils import Timer


class ModernHopfield(nn.Module):
    """Ramsauer et al 2020"""
    def __init__(self, input_size, hidden_size, num_steps=1):
        super().__init__()
        self.input_size = input_size # M
        self.hidden_size = hidden_size # N

        self.num_steps = num_steps
        self.mem = nn.Parameter(torch.zeros(self.hidden_size, self.input_size)) #[N,M]
        self.beta = nn.Parameter(torch.tensor(1.))

        self.reset()

    def reset(self):
        nn.init.xavier_normal_(self.mem)

    def forward(self, input):
        state = input #[B,M,1]
        for step in range(self.num_steps): #TODO: continuous dynamics, ensure it gets to FP (verify w many time steps)
            state = self.update_state(state) #[B,M,1]
        return state

    def update_state(self, state):
        h_ = torch.matmul(self.mem, state) #[N,M],[B,M,1]->[B,N,1]
        h = F.softmax(self.beta*h_, dim=1) #[B,N,1]
        state = torch.matmul(self.mem.t(), h) #[M,N],[B,N,1]->[B,M,1]
        return state


class AutoassociativeDataset(Dataset):
    """
    Convert a dataset with (input, target) pairs to work with autoassociative
    memory networks, returning (aa_input, aa_target) pairs, where
    aa_input = [input, init] and aa_target = [input, target]
    """
    def __init__(self, dataset, output_init_value=0):
        self.dataset = dataset
        self.output_init_value = output_init_value
        self.input_size = dataset[0][0].numel() # Mv
        self.target_size = dataset[0][1].numel() # Mc

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        output_init = torch.full_like(target, self.output_init_value)
        aa_input = torch.cat((input, output_init)) # M=Mv+Mc
        aa_target = torch.cat((input, target)) # M=Mv+Mc
        return aa_input, aa_target

    def __len__(self):
        return len(self.dataset)


def autoassociative_loss(aa_output, aa_target, num_outputs=10, loss_fn=nn.MSELoss()):
    output = aa_output[:, -num_outputs:, :] #[B,M,1]->[B,Mc,1]
    target = aa_target[:, -num_outputs:, :] #[B,M,1]->[B,Mc,1]
    loss = loss_fn(output, target)
    return loss


def autoassociative_acc(aa_output, aa_target, num_outputs=10):
    output = aa_output[:, -num_outputs:, :] #[B,M,1]->[B,Mc,1]
    target = aa_target[:, -num_outputs:, :] #[B,M,1]->[B,Mc,1]
    output_class = torch.argmax(output, dim=1)
    target_class = torch.argmax(target, dim=1)
    acc = (output_class==target_class).float().mean()
    return acc


def train(net, train_loader, test_loader=None, epochs=10, print_every=100, device='cpu'):
    optimizer = torch.optim.Adam(net.parameters())

    logger = defaultdict(list)
    iteration = 0
    for epoch in range(1,epochs+1):
        for batch_num, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)

            iteration += 1

            output = net(input)
            loss = autoassociative_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % print_every == 0:
                acc = autoassociative_acc(output, target)
                logger['iter'].append(iteration)
                logger['train_loss'].append(loss.item())
                logger['train_acc'].append(acc.item())

                log_str = 'iter={:3d}({:5d}) train_loss={:.3f} train_acc={:.2f}' \
                             .format(epoch, iteration, loss, acc)

                if test_loader is not None:
                    with torch.no_grad():
                        test_input, test_target = next(iter(test_loader))
                        test_input, test_target = test_input.to(device), test_target.to(device)

                        test_output = net(test_input)
                        test_loss = autoassociative_loss(test_output, test_target)
                        test_acc = autoassociative_acc(test_output, test_target)

                    logger['test_loss'].append(test_loss.item())
                    logger['test_acc'].append(test_acc.item())
                    log_str += ' test_loss={:.3f} test_acc={:.2f}'.format(test_loss, test_acc)
                print(log_str)
    return logger


#%%
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device =', device)

    #get dataset
    to_vec = transforms.Compose([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: x.view(-1,1))])
    to_onehot = transforms.Lambda(lambda y: torch.zeros(10,1)
                                  .scatter_(0, torch.tensor([[y]]), value=1))
    train_data = AutoassociativeDataset(
                    datasets.MNIST(root='./data/', download=True,
                                   transform=to_vec, target_transform=to_onehot)
                    )
    test_data = AutoassociativeDataset(
                    datasets.MNIST(root='./data/', train=False, download=True,
                                   transform=to_vec, target_transform=to_onehot)
                    )

    # train_data.dataset.data.to(device)
    # test_data.dataset.data.to(device)

    #define network
    input_size = train_data[0][0].numel()
    hidden_size = 1000
    net = ModernHopfield(input_size, hidden_size, num_steps=1)

    #train network
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    with Timer(device):
        net.to(device)
        logger = train(net, train_loader, test_loader, epochs=10, device=device)

    #%%plot
    fig, ax = plt.subplots(2,1)
    ax[0].plot(logger['iter'], logger['train_loss'], label='Train')
    ax[0].plot(logger['iter'], logger['test_loss'], label='Test')
    ax[0].set_ylabel('Loss (MSE)')
    ax[0].set_title('ModernHopfield (N={}), MNIST (B={})'.format(hidden_size, batch_size))
    ax[0].legend()

    ax[1].plot(logger['iter'], logger['train_acc'])
    ax[1].plot(logger['iter'], logger['test_acc'])
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Accuracy')

    #TODO: plot weights (reshape to MNIST shape)
    #TODO: get to 98%
