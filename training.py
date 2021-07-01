from collections import defaultdict
import warnings

import torch
from torch import nn
import torch.nn.functional as F


class AssociativeTrain():
    def __init__(self, net, train_loader, test_loader=None, only_class_loss=True, logger=None, print_every=100):
        self.name = None
        self.net = net
        self.device = next(net.parameters()).device  # assume all model params on same device

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_class_units = train_loader.dataset.target_size #Mc
        self.only_class_loss = only_class_loss

        self.print_every = print_every
        if logger is None:
            self.logger = defaultdict(list)
            self.iteration = 0
        else:
            self.logger = logger
            self.iteration = logger['iter'][-1]


    def _update_parameters(self, output, target):
        raise NotImplementedError


    def loss_fn(self, output, target):
        if self.only_class_loss:
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


    def __call__(self, epochs=10):
        print(f'Training: {self.name}')
        for epoch in range(1, epochs+1):
            for batch_num, (input, target) in enumerate(self.train_loader):
                self.iteration += 1

                input, target = input.to(self.device), target.to(self.device)
                output = self.net(input)

                self._update_parameters(output, target)

                if self.iteration % self.print_every == 0:
                    loss = self.loss_fn(output, target) #TODO this is getting computed twice in SGD
                    acc = self.acc_fn(output, target)
                    self.logger['iter'].append(self.iteration)
                    self.logger['train_loss'].append(loss.item())
                    self.logger['train_acc'].append(acc.item())

                    log_str = 'ep={:3d} it={:5d} loss={:.4f} acc={:.3f}' \
                                 .format(epoch, self.iteration, loss, acc)

                    if self.test_loader is not None:
                        with torch.no_grad():
                            test_input, test_target = next(iter(self.test_loader))
                            test_input, test_target = test_input.to(self.device), test_target.to(self.device)

                            test_output = self.net(test_input)
                            test_loss = self.loss_fn(test_output, test_target)
                            test_acc = self.acc_fn(test_output, test_target)

                        self.logger['test_loss'].append(test_loss.item())
                        self.logger['test_acc'].append(test_acc.item())
                        log_str += ' test_loss={:.4f} test_acc={:.3f}'.format(test_loss, test_acc)
                    print(log_str)
        return dict(self.logger)



class SGDTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, **kwargs):
        super().__init__(net, train_loader, **kwargs)
        self.name = 'SGD'
        self.optimizer = torch.optim.Adam(net.parameters())


    def _update_parameters(self, output, target):
        loss = self.loss_fn(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



class FPTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, lr=0.1, **kwargs):
        super().__init__(net, train_loader, **kwargs)
        self.name = 'FPT'
        self.lr = lr

        #parameters
        self.W = net.W
        self.beta = net.beta


    def __call__(self, epochs=10):
        with torch.no_grad():
            super().__call__(epochs)


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

        dLdW = self._dLdW(output, target)

        # dLdW_nb = compute_dLdW_nobatch(output.squeeze(0), target.squeeze(0),
        #                               net.W, net.beta)
        # assert (dLdW-dLdW_nb).abs().mean() < 1e-10

        self.W -= self.lr*dLdW
        del dLdW

        # in principle should not need this since it only releases memory to *outside*
        # programs but in practice helps with OOM (perhaps due to PyTorch mem leaks?)
        torch.cuda.empty_cache()


    def _dLdW(self, v, t):
        """
        Assumes:
            Modern Hopfield dynamics, tau*dv/dt = -v + W*softmax(W^T*v)
            v is a fixed point, dv/dt = 0, i.e. v = W*softmax(W^T*v)
            MSE loss, L = 1/2B sum_i sum_b (v_i - t_i)^2, where B is batch size
        """
        N,M = self.W.shape
        g = v  # [B,M,1]
        h = self.W@g  # [N,M]@[B,M,1] --> [B,N,1]
        f = F.softmax(self.beta*h, dim=1)  # [B,N,1]

        if self.only_class_loss:
            e = v[:, -self.n_class_units:] - t[:, -self.n_class_units:]  # error [B,Mc,1]
        else:
            e = v-t #[B,M,1]
        del h, v, t  # free up memory

        # ([B,N,N]-[B,N,1]@[B,1,N])@[N,M] --> [B,N,M]
        bFFW = self.beta * (f.squeeze().diag_embed() - f @ f.transpose(1, 2)) @ self.W
        A = torch.eye(M, device=self.device) - self.W.t() @ bFFW  # [M,M]+[M,N]@[B,N,M] --> [B,M,M]

        if self.only_class_loss:  # only compute loss over the last Mc units, M=Md+Mc
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
        N,M = self.W.shape

        f = F.softmax(self.beta * self.W @ v, dim=0)
        bFFW = self.beta * (torch.diag(f[:, 0]) - f @ f.T) @ self.W
        A = torch.eye(M, device=self.device) - self.W.T @ bFFW
        A_inv = torch.linalg.inv(A)
        a = A_inv[:, -self.n_class_units:] @ (v-t)[-self.n_class_units:]
        dLdW = f@ a.T + bFFW @ a @ v.T
        return dLdW



#################
# Miscellaneous #
#################
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
