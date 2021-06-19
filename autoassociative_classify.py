#built-in
from collections import defaultdict

#third-party
import torch
from torch import nn

#custom
import plots
import data
from utils import Timer
from networks import ModernHopfield
from training import SGD_train, fixed_point_train

#??? Train on one step --> probably won't extrapolate to many
#??? Try "full" autoassociative task
#??? Can we remove the "dead" units?
#??? Normalize each weight vector per hidden unit to 1 --> removes the single giant hidden unit?

#%% train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

#get dataset
train_data, test_data = data.get_aa_mnist_classification_data(include_test=True,
                                                              downsample=5)

#define network
input_size = train_data[0][0].numel()
hidden_size = 50
net = ModernHopfield(input_size, hidden_size, dt=0.05, num_steps=10/.05, tau=1)
logger = None

#%%train network
batch_size = 1
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100)
with Timer(device):
    net.to(device)
    # logger = SGD_train(net, train_loader, test_loader, logger=logger, epochs=300)
    logger = fixed_point_train(net, train_loader, test_loader, logger=logger, epochs=300)


#%% plot
plots.plot_loss_acc(logger['train_loss'], logger['train_acc'],
              logger['test_loss'], logger['test_acc'], logger['iter'],
              title='ModernHopfield (N={}), MNIST (B={})'.format(hidden_size, batch_size))

plots.plot_weights_mnist(net.mem)

n_per_class = 10
debug_data = data.AutoassociativeDataset(data.filter_classes(test_data.dataset,n_per_class=n_per_class))
debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))
state_debug_history = net(debug_input, debug=True)
plots.plot_hidden_max_argmax(state_debug_history, n_per_class)

#%%
n_per_class = 2
debug_data = data.AutoassociativeDataset(data.filter_classes(test_data.dataset,n_per_class=n_per_class))
debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))
num_steps_train = net.num_steps
net.num_steps = 1000
state_debug_history = net(debug_input, debug=True)
plots.plot_state_update_magnitude(state_debug_history, n_per_class, num_steps_train)
