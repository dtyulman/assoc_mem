#built-in
from collections import defaultdict

#third-party
import torch
import torch.nn.functional as F
from torch import nn

#custom
from utils import Timer
import plots
import data


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


def train(net, train_loader, test_loader=None, epochs=10, print_every=100):
    device = next(net.parameters()).device #assumes all model params on same device
    optimizer = torch.optim.Adam(net.parameters())
    logger = defaultdict(list)

    iteration = 0
    for epoch in range(1,epochs+1):
        for batch_num, (input, target) in enumerate(train_loader):
            iteration += 1

            input, target = input.to(device), target.to(device)
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
    return dict(logger)


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

#get dataset
train_data, test_data = data.get_aa_mnist_classification_data(include_test=True)

#define network
input_size = train_data[0][0].numel()
hidden_size = 100
net = ModernHopfield(input_size, hidden_size, num_steps=1)

#train network
batch_size = 500
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

with Timer(device):
    net.to(device)
    logger = train(net, train_loader, test_loader, epochs=30)

#%%plot
plots.plot_loss_acc(logger['train_loss'], logger['train_acc'],
              logger['test_loss'], logger['test_acc'], logger['iter'],
              title='ModernHopfield (N={}), MNIST (B={})'.format(hidden_size, batch_size))

plots.plot_weights_mnist(net.mem)

#TODO: get to 98%
