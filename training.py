from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F


################
# SGD Training #
################
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


def SGD_train(net, train_loader, test_loader=None, epochs=10, logger=None, print_every=100):
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
