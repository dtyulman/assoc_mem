from collections import defaultdict
import warnings
import time

import torch
from torch import nn
import torch.nn.functional as F


class AssociativeTrain():
    def __init__(self, net, train_loader, test_loader=None, lr=0.001, lr_decay=1.,
                 loss_mode='class', logger=None, print_every=100, sparse_logging_factor=10):#, **kwargs):
        #kwargs ignores any extra values passed in via a Config
        self.name = None
        self.net = net

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_class_units = train_loader.dataset.target_size #Mc

        assert loss_mode in ['class', 'full'], f"Invalid loss mode: '{loss_mode}'"
        self.loss_mode = loss_mode

        self.lr = lr
        self.lr_decay = lr_decay

        self.print_every = print_every
        self.sparse_logging_factor = sparse_logging_factor
        if logger is None:
            self.logger = defaultdict(list)
            self.iteration = 0
        else:
            self.logger = logger
            self.iteration = logger['iter'][-1]


    def _update_parameters(self, output, target):
        raise NotImplementedError


    def loss_fn(self, output, target):
        if self.loss_mode == 'class':
            output = output[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
            target = target[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
        loss = F.mse_loss(output, target, reduction='mean')
        return loss


    def acc_fn(self, output, target):
        output = output[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
        target = target[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
        output_class = torch.argmax(output, dim=1)
        target_class = torch.argmax(target, dim=1)
        acc = (output_class == target_class).float().mean()
        return acc


    def __call__(self, epochs=10, label=''):
        print(f'Training: {self.name}' + (f', {label}' if label else ''))
        self.device = next(self.net.parameters()).device  # assume all model params on same device

        with Timer():
            for epoch in range(1, epochs+1):
                for batch_num, (input, target) in enumerate(self.train_loader):
                    self.iteration += 1

                    input, target = input.to(self.device), target.to(self.device)
                    output = self.net(input)

                    self._update_parameters(output, target)

                    #logging
                    if self.iteration % self.print_every == 0:
                        #params / param stats
                        for param_name, param_value in self.net.named_parameters():
                            if param_value.numel() == 1:
                                self.logger[param_name].append(param_value.item())
                            else:
                                self.logger[param_name+'/mean'].append(param_value.mean().item())
                                self.logger[param_name+'/std'].append(param_value.std().item())
                                if self.iteration % (self.sparse_logging_factor*self.print_every) == 0:
                                    #log full parameter more sparsely to save space
                                    self.logger[param_name].append(param_value.detach().cpu().numpy())

                        #loss/acc
                        loss = self.loss_fn(output, target) #TODO this is getting computed twice in SGD
                        acc = self.acc_fn(output, target)
                        self.logger['iter'].append(self.iteration)
                        self.logger['train_loss'].append(loss.item())
                        self.logger['train_acc'].append(acc.item())

                        log_str = 'ep={:3d} it={:5d} loss={:.5f} acc={:.3f}' \
                                     .format(epoch, self.iteration, loss, acc)

                        #test loss/acc if applicable
                        if self.test_loader is not None:
                            with torch.no_grad():
                                test_input, test_target = next(iter(self.test_loader))
                                test_input, test_target = test_input.to(self.device), test_target.to(self.device)

                                test_output = self.net(test_input)
                                test_loss = self.loss_fn(test_output, test_target)
                                test_acc = self.acc_fn(test_output, test_target)

                            self.logger['test_loss'].append(test_loss.item())
                            self.logger['test_acc'].append(test_acc.item())
                            log_str += ' test_loss={:.5f} test_acc={:.3f}'.format(test_loss, test_acc)
                        print(log_str)

                if self.lr_decay:
                    self.lr *= self.lr_decay
        return dict(self.logger)



class SGDTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, **kwargs):
        #explicit test_loader kwarg also allows it to be passed as positional arg
        super().__init__(net, train_loader, test_loader, **kwargs)
        self.name = 'SGD'
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)


    def _update_parameters(self, output, target):
        loss = self.loss_fn(output, target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if self.iteration % self.print_every == 0:
            grad_norm = torch.cat([p.reshape(-1,1) for p in self.net.parameters() if p.requires_grad]).norm().item()
            self.logger['grad_norm'].append(grad_norm)



class FPTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, lr=0.1, **kwargs):
        #explicit test_loader kwarg also allows it to be passed as positional arg
        #modified default lr
        super().__init__(net, train_loader, test_loader, lr=lr, **kwargs)
        self.name = 'FPT'


    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return super().__call__(*args, **kwargs)


    def _update_parameters(self, output, target):
        # dvdW = compute_dvdW(output, net.W, net.beta) #[B,(N,M),M,1]

        # #only consider the neurons reporting the class
        # output_class = output[:, -num_classes:, :] #[B,M,1]->[B,Mc,1]
        # target_class = target[:, -num_classes:, :] #[B,M,1]->[B,Mc,1]
        # dvdW_class = dvdW[:,:,:, -num_classes:, :] #[B,(N,M),M,1] --> #[B,(N,M),Mc,1]

        # #assume MSE loss
        # error_class = (output_class - target_class).unsqueeze(1).unsqueeze(1) #[B,Mc,1]-->[B,1,1,Mc,1]
        # dLdW1 = (error_class*dvdW_class).squeeze().sum(dim=0).sum(dim=-1) #[B,(N,M),Mc,1]-->[N,M]
        # del dvdW, output_class, target_class, dvdW_class, error_class, dLdW

        dLdW = self._dLdW(output, target) #TODO: put into W.grad?

        # dLdW_nb = compute_dLdW_nobatch(output.squeeze(0), target.squeeze(0),
        #                               net.W, net.beta)
        # assert (dLdW-dLdW_nb).abs().mean() < 1e-10

        if self.net.normalize_weight:
            #W is normalized, _W is not, need one more step of chain rule
            Z = self.net._W.norm(dim=1, keepdim=True) #[N,M]->[N,1]
            S = (dLdW * self.net._W).sum(dim=1, keepdim=True) #[N,M],[N,M]->[N,1]
            dLd_W = dLdW / Z - self.net._W *(S / Z**3)
            self.net._W -= self.lr * dLd_W
        else: #W==_W, so dLd_W[u,i]==dLdW[u,i]
            self.net._W -= self.lr * dLdW

        if self.iteration % self.print_every == 0:
            grad_norm = dLdW.norm().item()
            self.logger['grad_norm'].append(grad_norm)

        #in principle should not need this since it only releases memory to *outside*
        #programs but in practice helps with OOM (perhaps due to PyTorch mem leaks?)
        del dLdW
        torch.cuda.empty_cache()



    def _dLdW(self, v, t):
        """
        Assumes:
            Modern Hopfield dynamics, tau*dv/dt = -v + W*softmax(W^T*v)
            v is a fixed point, dv/dt = 0, i.e. v = W*softmax(W^T*v)
            MSE loss, L = 1/2B sum_i sum_b (v_i - t_i)^2, where B is batch size
        """
        N,M = self.net.W.shape
        g = v  # [B,M,1]
        h = self.net.W@g  # [N,M]@[B,M,1] --> [B,N,1]
        f = F.softmax(self.net.beta*h, dim=1)  # [B,N,1]

        if self.loss_mode == 'class':
            e = v[:, -self.n_class_units:] - t[:, -self.n_class_units:]  # error [B,Mc,1]
        else:
            e = v-t #[B,M,1]
        del h, v, t  # free up memory

        # ([B,N,N]-[B,N,1]@[B,1,N])@[N,M] --> [B,N,M]
        bFFW = self.net.beta * (f.squeeze().diag_embed() - f @ f.transpose(1, 2)) @ self.net.W
        A = torch.eye(M, device=self.device) - self.net.W.t() @ bFFW  # [M,M]+[M,N]@[B,N,M] --> [B,M,M]

        if self.loss_mode == 'class': # only compute loss over the last Mc units, M=Md+Mc
            Ainv = torch.linalg.inv(A)  # [B,M,M]
            a = Ainv[:, -self.n_class_units:].transpose(1, 2) @ e #[B,M,Mc]@[B,Mc,1]--> [B,M,1]
            del Ainv
        else:
            # a = A^(-1)^T * e <=> a = A^(-1) * e  b/c A and A^(-1) symmetric for g(x)=x
            a = torch.linalg.solve(A.transpose(1, 2), e)  # [B,M,M],[B,M,1] --> [B,M,1]
        del A, e

        # dLdW = f @ a.transpose(1,2) + bFFW @ a @ g.transpose(1,2) #[B,N,1]@[B,1,Mc]+[B,N,M]@[B,M,1]@[B,1,M] --> [B,N,M]
        dLdW = bFFW @ a @ g.transpose(1, 2);
        del bFFW, g
        dLdW = dLdW + f @ a.transpose(1, 2)
        del f, a

        return dLdW.mean(dim=0)


    def _dLdW_nobatch(self, v, t): #for debugging
        assert(len(v.shape) == 2), 'output v cannot be batched'
        assert(len(t.shape) == 2), 'target t cannot be batched'
        N,M = self.net.W.shape

        f = F.softmax(self.net.beta * self.net.W @ v, dim=0)
        bFFW = self.net.beta * (torch.diag(f[:, 0]) - f @ f.T) @ self.net.W
        A = torch.eye(M, device=self.device) - self.net.W.T @ bFFW
        A_inv = torch.linalg.inv(A)
        a = A_inv[:, -self.n_class_units:] @ (v-t)[-self.n_class_units:]
        dLdW = f@ a.T + bFFW @ a @ v.T
        return dLdW


###########
# Helpers #
###########
class Timer():
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        print('Starting timer{}'.format(f': {self.name}...' if self.name is not None else '...'))
        self.start_time = time.perf_counter()

    def __exit__(self, *args):
        stop_time = time.perf_counter()
        elapsed = stop_time - self.start_time
        elapsed_str = f'Time elapsed: {elapsed}'
        if self.name is not None:
            elapsed_str = f'{self.name}: {elapsed_str} sec'
        print(elapsed_str)


class Logger():
    #TODO: tensorboard
    def __init__(self, folder=None):
        raise NotImplementedError()
        self.folder = folder
        self.log = defaultdict(list)

    def write(self, key, value, iteration):
        pass


############
# Not used #
############
class MPELoss(nn.modules.loss._Loss):
    """Like MSELoss, but takes the P power instead of Square. If P odd, takes absolute value first
    i.e. L = 1/N sum |x-y|^P where N = x.numel()
    """
    def __init__(self, P=1, reduction='mean'):
        super().__init__(reduction=reduction)
        self.P = 1

    def forward(self, input, target):
        assert input.shape == target.shape, 'Input and target sizes must be the same'
        loss=(input-target).abs()**self.p
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class NoBatchWarning(RuntimeWarning):
    pass
warnings.simplefilter('once', NoBatchWarning)

def compute_dvdW(v, W, beta):
    """This is specialized for Modern Hopfield with f(h)=softmax(beta*h) and g(v)=v. Easily
    generalizable to any elementwise g_i(v) = g(v_i). Need to do more work for arbitrary
    g_i(v1..vM) or f_i(h1..hN).
    """
    dev = W.device

    N, M = W.shape
    g = v  # [B,M,1]
    h = torch.matmul(W, g)  # [B,N,1]
    f = F.softmax(beta*h, dim=1)  # [B,N,1]
    del h, v  # free up memory

    # for a single synaptic weight from v_i to h_u: A*dvdW_{ui} = b_{ui}, [M,M]@[M,1]-->[M,1]
    # for all the synaptic weights: A*dvdW = b, shape [M,M]@[(N,M),M,1]-->[(N,M),M,1]
    # batched: A*dvdW = b, [B,M,M]@[B,(N,M),M,1]-->[B,(N,M),M,1]

    # Want to do this, but it uses too much memory:
    # ffT = torch.bmm(f, f.transpose(1,2)) #[B,N,1],[B,1,N]-->[B,N,N]
    # Df = f.squeeze().diag_embed() #[B,N,1]-->[B,N,N]
    # A = torch.eye(M, device=dev) + beta * W.t() @ (Df-ffT) @ W #[M,M] + 1*[M,N]@[B,N,N]@[N,M]-->[B,M,M]
    #
    # DDf = f.expand(-1,-1,M).diag_embed().unsqueeze(-1) #[B,N,1]-->[B,N,M]-->[B,(N,M),M]-->[B,(N,M),M,1]
    # fgT = torch.bmm(f, g.transpose(1,2)).unsqueeze(-1).unsqueeze(-1) #[B,N,1]@[B,1,M]-->[B,(N,M)]-->[B,(N,M),1,1]
    # WTf = (W.t() @ f).unsqueeze(1).unsqueeze(1) #[M,N]@[B,N,1]-->[B,M,1]-->[B,(1,1),M,1]
    # WT_ = W.tile(1,M).view(1,N,M,M,1) #[N,M]-->[N,M*M]-->[1,(N,M),M,1]
    # b = DDf + beta*fgT*(WT_-WTf) #[B,(N,M),M,1] via broadcasting

    A = f.squeeze().diag_embed()  # Df
    A = A - torch.bmm(f, f.transpose(1, 2))  # Df-ffT
    A = beta*W.t() @ A  # beta*W.t() @ (Df-ffT) #note minor numerical difference swapping this and next line
    A = A @ W  # beta*W.t() @ (Df-ffT) @ W
    A = A + torch.eye(M, device=dev)  # eye + beta*W.t()@(Df-ffT)@W

    b = W.tile(1, M).view(1, N, M, M, 1)  # WT_
    b = b - (W.t()@f).unsqueeze(1).unsqueeze(1)  # WT_-WTf
    b = b * beta*torch.bmm(f, g.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1)  # beta*fgT*(WT_-WTf)
    b = b + f.expand(-1, -1, M).diag_embed().unsqueeze(-1)  # DDf + beta*fgT*(WT_-WTf)

    # Want to do this, but it's slower (because inverts A for each b_{ui}?)
    # dvdW = torch.linalg.solve(A.unsqueeze(1).unsqueeze(1), b)
    Ainv = (torch.linalg.inv(A)).unsqueeze(1).unsqueeze(1)  # [B,1,1,M,M]
    del A

    try:
        dvdW = Ainv @ b  # [B,1,1,M,M],[B,(N,M),M,1]-->[B,(N,M),M,1]
    except RuntimeError as e:  # out of memory
        warnings.warn(
            f'Cannot batch multiply: "{e.args[0]}." Looping over batch...', NoBatchWarning)
        dvdW = torch.empty_like(b)
        for batch_idx, (Ainv_, b_) in enumerate(zip(Ainv, b)):
            dvdW[batch_idx] = Ainv_ @ b_

    return dvdW
