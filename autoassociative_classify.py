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
    def __init__(self, input_size, hidden_size, beta=None, tau=None,
                 num_steps=None, dt=1, fp_threshold=0.001):
        super().__init__()
        self.input_size = input_size # M
        self.hidden_size = hidden_size # N

        self.mem = nn.Parameter(torch.zeros(self.hidden_size, self.input_size)) #[N,M]
        nn.init.xavier_normal_(self.mem)

        self.beta = nn.Parameter(torch.tensor(beta or 1.), requires_grad=(beta is None))
        self.tau = nn.Parameter(torch.tensor(tau or 1.), requires_grad=(tau is None))

        assert dt<=1, 'Step size dt should be <=1'
        self.dt = dt
        self.num_steps = int(1/self.dt) if num_steps is None else num_steps
        self.fp_threshold = fp_threshold


    def forward(self, input, debug=False):
        state = input #[B,M,1]
        if debug:
            state_debug_history = []
        update_magnitude = float('inf')
        step = 0
        while update_magnitude > self.fp_threshold and step < self.num_steps:
            prev_state = state
            state = self.update_state(prev_state, debug=debug) #[B,M,1]
            if debug: #state is actually state_debug
                state_debug_history.append(state) #store it
                state = state['state'] #extract the actual state from the dict
            with torch.no_grad():
                update_magnitude = (prev_state-state).norm()
            step += 1
        if debug:
            return state_debug_history
        return state


    def update_state(self, state, debug=False):
        """
        Continuous: tau*dv/dt = -v + X*softmax(X^T*v)
        Discretized: v(t+dt) = v(t) + dt/tau [-v(t) + X*softmax(X^T*v(t))]
        """
        h_ = torch.matmul(self.mem, state) #[N,M],[B,M,1]->[B,N,1]
        h = F.softmax(self.beta*h_, dim=1) #[B,N,1]
        memT_h = torch.matmul(self.mem.t(), h) #[M,N],[B,N,1]->[B,M,1]
        state = state + (self.dt/self.tau)*(memT_h - state) #[B,M,1]
        if debug:
            state_debug = {}
            for key in ['h_', 'h', 'memT_h', 'state']:
                state_debug[key] = locals()[key].detach()
            return state_debug
        return state


class MPELoss(nn.modules.loss._Loss):
    """Like MSELoss, but takes the P power instead of Square. If P odd, takes absolute value first
    i.e. L = 1/N sum |x-y|^P where N = x.numel()
    """
    def __init__(self, P=1, reduction='mean'):
        super().__init__(reduction=reduction)
        self.P = 1

    def forward(self, input, target):
        assert input.shape == target.shape, 'Input and target sizes must be the same'
        loss = (input-target).abs()**self.p
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss= loss.mean()
        return loss


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


def train(net, train_loader, test_loader=None, epochs=10, logger=None, print_every=100):
    device = next(net.parameters()).device #assumes all model params on same device
    optimizer = torch.optim.Adam(net.parameters())
    if logger is None:
        logger = defaultdict(list)
        iteration = 0
    else:
        iteration = logger['iter'][-1]

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


#%% train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

#get dataset
train_data, test_data = data.get_aa_mnist_classification_data(include_test=True)

#define network
input_size = train_data[0][0].numel()
hidden_size = 1000
net = ModernHopfield(input_size, hidden_size, dt=0.05, num_steps=10/.05)
logger = None

#train network
batch_size = 100
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
with Timer(device):
    net.to(device)
    logger = train(net, train_loader, test_loader, logger=logger, epochs=300)


#%% plot
plots.plot_loss_acc(logger['train_loss'], logger['train_acc'],
              logger['test_loss'], logger['test_acc'], logger['iter'],
              title='ModernHopfield (N={}), MNIST (B={})'.format(hidden_size, batch_size))

plots.plot_weights_mnist(net.mem)

n_per_class = 10
debug_data = data.AutoassociativeDataset(data.filter_classes(test_data.dataset,n_per_class=n_per_class))
debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))
state_debug_history = net(debug_input, debug=True)
plots.plot_hidden_max(state_debug_history, n_per_class)

n_per_class = 2
debug_data = data.AutoassociativeDataset(data.filter_classes(test_data.dataset,n_per_class=n_per_class))
debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))
max_train_steps = net.num_steps
net.num_steps = 1000
state_debug_history = net(debug_input, debug=True)
plots.plot_state_update_magnitude(state_debug_history, n_per_class, max_train_steps=max_train_steps)
