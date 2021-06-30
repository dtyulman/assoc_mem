from collections import defaultdict
import warnings

import torch
from torch import nn
import torch.nn.functional as F

class NoBatchWarning(RuntimeWarning):
    pass
warnings.simplefilter('once', NoBatchWarning)


########################
# Fixed point training #
########################
def fixed_point_train(net, train_loader, test_loader=None, lr=0.001, epochs=10, logger=None, print_every=100):
    print('Training: fixed point optimization')
    device = next(net.parameters()).device #assumes all model params on same device
    num_classes = 10 #assume MNIST

    if logger is None:
        logger = defaultdict(list)
        iteration = 0
    else:
        iteration = logger['iter'][-1]

    for epoch in range(1,epochs+1):
        for batch_num, (input, target) in enumerate(train_loader):
            iteration += 1

            with torch.no_grad():
                input, target = input.to(device), target.to(device)
                output = net(input) #[B,M,1]

                # dvdW = compute_dvdW(output, net.W, net.beta) #[B,(N,M),M,1]

                # #only consider the neurons reporting the class
                # output_class = output[:, -num_classes:, :] #[B,M,1]->[B,Mc,1]
                # target_class = target[:, -num_classes:, :] #[B,M,1]->[B,Mc,1]
                # dvdW_class = dvdW[:,:,:, -num_classes:, :] #[B,(N,M),M,1] --> #[B,(N,M),Mc,1]

                # #assume MSE loss
                # error_class = (output_class - target_class).unsqueeze(1).unsqueeze(1) #[B,Mc,1]-->[B,1,1,Mc,1]
                # dLdW1 = (error_class*dvdW_class).squeeze().sum(dim=0).sum(dim=-1) #[B,(N,M),Mc,1]-->[N,M]
                # del dvdW, output_class, target_class, dvdW_class, error_class, dLdW

                dLdW = compute_dLdW(output, target, net.W, net.beta, Mc=num_classes)

                # dLdW_nb = compute_dLdW_nobatch(output.squeeze(0), target.squeeze(0),
                #                               net.W, net.beta)
                # assert (dLdW-dLdW_nb).abs().mean() < 1e-10

                net.W -= lr*dLdW
                del dLdW


                #in principle should not need this since it only releases memory to *outside*
                #programs but in practice helps with OOM (perhaps due to PyTorch mem leaks?)
                torch.cuda.empty_cache()

                if iteration % print_every == 0:
                    acc = autoassociative_acc(output, target)
                    loss = autoassociative_loss(output, target)
                    logger['iter'].append(iteration)
                    logger['train_loss'].append(loss.item())
                    logger['train_acc'].append(acc.item())

                    log_str = 'iter={:3d}({:5d}) train_loss={:.4f} train_acc={:.3f}' \
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
                        log_str += ' test_loss={:.4f} test_acc={:.3f}'.format(test_loss, test_acc)
                    print(log_str)
    return dict(logger)


def compute_dLdW_nobatch(v, t, KS, beta, Mc=10):
    assert(len(v.shape)==2), 'output v cannot be batched'
    assert(len(t.shape)==2), 'target t cannot be batched'
    N,M = KS.shape
    f = F.softmax(beta * KS @ v, dim=0)

    FF = torch.diag(f[:,0]) - f @ f.T
    A = torch.eye(M, device=KS.device) - beta * KS.T @ FF @ KS
    A_inv = torch.linalg.inv(A)
    a = A_inv[:,-Mc:] @ (v-t)[-Mc:]
    dKS1 = f @ a.T
    dKS2 = beta * FF @ KS @ a @ v.T
    return dKS1+dKS2


def compute_dLdW(v, t, W, beta, Mc=None):
    """
    Assumes:
        Modern Hopfield dynamics, tau*dv/dt = -v + W*softmax(W^T*v)
        v is a fixed point, dv/dt = 0, i.e. v = W*softmax(W^T*v)
        MSE loss, L = 1/2 sum_i (v_i - t_i)^2
    """
    N,M = W.shape
    g = v #[B,M,1]
    h = W@g #[N,M]@[B,M,1] --> [B,N,1]
    f = F.softmax(beta*h, dim=1) #[B,N,1]

    e = v-t  #error [B,M,1]
    del h, v, t #free up memory

    bFFW = beta*(f.squeeze().diag_embed() - f @ f.transpose(1,2)) @ W #([B,N,N]-[B,N,1]@[B,1,N])@[N,M] --> [B,N,M]
    A = torch.eye(M, device=W.device) - W.t() @ bFFW #[M,M]+[M,N]@[B,N,M] --> [B,M,M]

    if Mc: #only compute loss over the last Mc units
        Ainv = torch.linalg.inv(A) #[B,M,M]
        a = Ainv[:,-Mc:].transpose(1,2) @ e[:,-Mc:] #[B,M,Mc]@[B,Mc,1]--> [B,M,1]
        del Ainv
    else:
        #a = A^(-1)^T * e <=> a = A^(-1) * e  b/c A and A^(-1) symmetric for g(x)=x
        a = torch.linalg.solve(A.transpose(1,2), e) #[B,M,M],[B,M,1] --> [B,M,1]
    del A, e

    # dLdW = f @ a.transpose(1,2) + bFFW @ a @ g.transpose(1,2) #[B,N,1]@[B,1,Mc]+[B,N,M]@[B,M,1]@[B,1,M] --> [B,N,M]
    dLdW = bFFW @ a @ g.transpose(1,2);
    del bFFW, g
    dLdW = dLdW + f @ a.transpose(1,2)
    del f, a
    return dLdW.mean(dim=0)


def compute_dvdW(v, W, beta):
    """This is specialized for Modern Hopfield with f(h)=softmax(beta*h) and g(v)=v. Easily
    generalizable to any elementwise g_i(v) = g(v_i). Need to do more work for arbitrary
    g_i(v1..vM) or f_i(h1..hN).
    """
    dev = W.device

    N,M = W.shape
    g = v #[B,M,1]
    h = torch.matmul(W, g) #[B,N,1]
    f = F.softmax(beta*h, dim=1) #[B,N,1]
    del h, v #free up memory

    # for a single synaptic weight from v_i to h_u: A*dvdW_{ui} = b_{ui}, [M,M]@[M,1]-->[M,1]
    # for all the synaptic weights: A*dvdW = b, shape [M,M]@[(N,M),M,1]-->[(N,M),M,1]
    # batched: A*dvdW = b, [B,M,M]@[B,(N,M),M,1]-->[B,(N,M),M,1]

    #Want to do this, but it uses too much memory:
    # ffT = torch.bmm(f, f.transpose(1,2)) #[B,N,1],[B,1,N]-->[B,N,N]
    # Df = f.squeeze().diag_embed() #[B,N,1]-->[B,N,N]
    # A = torch.eye(M, device=dev) + beta * W.t() @ (Df-ffT) @ W #[M,M] + 1*[M,N]@[B,N,N]@[N,M]-->[B,M,M]
    #
    # DDf = f.expand(-1,-1,M).diag_embed().unsqueeze(-1) #[B,N,1]-->[B,N,M]-->[B,(N,M),M]-->[B,(N,M),M,1]
    # fgT = torch.bmm(f, g.transpose(1,2)).unsqueeze(-1).unsqueeze(-1) #[B,N,1]@[B,1,M]-->[B,(N,M)]-->[B,(N,M),1,1]
    # WTf = (W.t() @ f).unsqueeze(1).unsqueeze(1) #[M,N]@[B,N,1]-->[B,M,1]-->[B,(1,1),M,1]
    # WT_ = W.tile(1,M).view(1,N,M,M,1) #[N,M]-->[N,M*M]-->[1,(N,M),M,1]
    # b = DDf + beta*fgT*(WT_-WTf) #[B,(N,M),M,1] via broadcasting

    A = f.squeeze().diag_embed() #Df
    A = A - torch.bmm(f, f.transpose(1,2)) #Df-ffT
    A = beta*W.t() @ A #beta*W.t() @ (Df-ffT) #note minor numerical difference swapping this and next line
    A = A @ W #beta*W.t() @ (Df-ffT) @ W
    A = A + torch.eye(M, device=dev) #eye + beta*W.t()@(Df-ffT)@W

    b = W.tile(1,M).view(1,N,M,M,1) #WT_
    b = b - (W.t()@f).unsqueeze(1).unsqueeze(1) #WT_-WTf
    b = b * beta*torch.bmm(f, g.transpose(1,2)).unsqueeze(-1).unsqueeze(-1) #beta*fgT*(WT_-WTf)
    b = b + f.expand(-1,-1,M).diag_embed().unsqueeze(-1) #DDf + beta*fgT*(WT_-WTf)

    #Want to do this, but it's slower (because inverts A for each b_{ui}?)
    # dvdW = torch.linalg.solve(A.unsqueeze(1).unsqueeze(1), b)
    Ainv = (torch.linalg.inv(A)).unsqueeze(1).unsqueeze(1) #[B,1,1,M,M]
    del A

    try:
        dvdW = Ainv @ b #[B,1,1,M,M],[B,(N,M),M,1]-->[B,(N,M),M,1]
    except RuntimeError as e: #out of memory
        warnings.warn(f'Cannot batch multiply: "{e.args[0]}." Looping over batch...', NoBatchWarning)
        dvdW = torch.empty_like(b)
        for batch_idx, (Ainv_, b_) in enumerate(zip(Ainv, b)):
            dvdW[batch_idx] = Ainv_ @ b_

    return dvdW


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


def sgd_train(net, train_loader, test_loader=None, epochs=10, logger=None, print_every=100):
    print('Training: BPTT')

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
