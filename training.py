#built-in
import warnings, time
from copy import deepcopy

#third party
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#custom
import plots, data


class AssociativeTrain():
    def __init__(self, net, train_loader, test_loader=None, optimizer=None,
                 loss_mode='class', loss_fn='mse', acc_mode='class', acc_fn='l0', reg_loss=None,
                 logger=None, logdir=None, print_every=100, sparse_log_factor=10, **kwargs):
        if kwargs:
            #kwargs ignores any extra values passed in (eg. via a Config object)
            warnings.warn(f'Ignoring unused AssociativeTrain kwargs: {kwargs}')

        self.name = None
        self.net = net
        self.set_device()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_class_units = train_loader.dataset.dataset.num_classes #Mc

        self.optimizer = torch.optim.Adam(net.parameters()) if optimizer is None else optimizer

        self._set_loss_mode(loss_mode, loss_fn)
        self._set_acc_mode(acc_mode, acc_fn)
        self.reg_loss = reg_loss

        self.print_every = print_every
        self.sparse_log_factor = sparse_log_factor

        if logger is None:
            self.logger = Logger(logdir)
            self.write_log() #log everything before training occurs
        else:
            self.logger = logger


    def _set_loss_mode(self, loss_mode, loss_fn):
        assert loss_mode in ['class', 'full'], f"Invalid loss mode: '{loss_mode}'"
        self.loss_mode = loss_mode

        if loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'cos':
            raise NotImplementedError
        elif loss_fn == 'bce':
            raise nn.BCELoss()
        else:
            raise ValueError(f"Invalid loss function: '{loss_fn}'")


    def _set_acc_mode(self, acc_mode, acc_fn):
        assert acc_mode in ['class', 'full'], f"Invalid accuracy mode: '{acc_mode}'"
        self.acc_mode = acc_mode

        if acc_fn == 'L0':
            self.acc_fn = l0_acc
        elif acc_fn == 'L1' or acc_fn == 'mae':
            self.acc_fn = l1_acc
        elif acc_fn == 'cls':
            self.acc_fn = classifier_acc
        else:
            raise ValueError(f"Invalid accuracy function: '{acc_fn}'")


    def __call__(self, epochs=10, label=''):
        self.set_device()
        self.net.train() #put net in 'training' mode e.g. enable dropout if applicable
        with Timer(f'{self.__class__.__name__}' + (f', {label}' if label else '')):
            for _ in range(epochs):
                self.logger.epoch += 1
                for batch in self.train_loader:
                    self.logger.iteration += 1

                    input, target, clamp_mask = self._prep_data(*batch)
                    output = self.net(input, clamp_mask)
                    self._update_parameters(output, target)

                    #bug/feature: logs the resulting loss/acc *after* the param update,
                    #not the loss which caused the update
                    self.write_log(output=output, target=target)

        self.net.eval() #put net back in 'evaluation' mode e.g. disable dropout

        return self.logger


    def _prep_data(self, input, target, perturb_mask):
        input, target = input.to(self.device), target.to(self.device)
        clamp_mask = ~perturb_mask if ('clamp' in self.net.input_mode) else None
        return input, target, clamp_mask


    def _update_parameters(self, output, target):
        raise NotImplementedError


    def compute_loss(self, output, target):
        if self.loss_mode == 'class':
            output = output[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
            target = target[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
        loss = self.loss_fn(output, target)

        if self.reg_loss is not None:
            loss = loss + self.reg_loss()

        return loss


    def compute_acc(self, output, target):
        if self.acc_mode == 'class':
            output = output[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
            target = target[:, -self.n_class_units:]  # [B,M,1]->[B,Mc,1]
        return self.acc_fn(output, target)


    def set_device(self):
        self.device = next(self.net.parameters()).device  #assume all model params on same device


    @torch.no_grad()
    def write_log(self, input=None, target=None, clamp_mask=None, output=None,
                  train_loss=None, train_acc=None):

        if self.logger.iteration % self.print_every == 0:
            self.net.eval()

            #training loss/acc, compute if needed
            if train_loss is None or train_acc is None:
                assert (input is not None and target is not None and output is     None) or \
                       (input is     None and target is not None and output is not None) or \
                       (input is     None and target is     None and output is     None), \
                        'Provide either input+target, output+target, xor none of them'
                if output is None:
                    if input is None:
                        batch = next(iter(self.train_loader))
                        input, target, clamp_mask = self._prep_data(*batch)
                    output = self.net(input, clamp_mask)

                if train_loss is None:
                    train_loss = self.compute_loss(output, target)
                if train_acc is None:
                    train_acc = self.compute_acc(output, target)

            self.logger('train/loss', train_loss)
            self.logger('train/acc', train_acc)
            log_str = 'ep={:3d} it={:5d} loss={:.5e} acc={:.3f}' \
                         .format(self.logger.epoch, self.logger.iteration, train_loss, train_acc)

            #test loss/acc if applicable
            if self.test_loader is not None:
                test_batch = next(iter(self.test_loader))
                test_input, test_target, test_clamp_mask = self._prep_data(*test_batch)

                test_output = self.net(test_input, test_clamp_mask)
                test_loss = self.compute_loss(test_output, test_target)
                test_acc = self.compute_acc(test_output, test_target)

                self.logger('test/loss', test_loss)
                self.logger('test/acc', test_acc)
                log_str += ' test_loss={:.5e} test_acc={:.3f}'.format(test_loss, test_acc)

            #params / param stats
            for param_name, param_value in self.net.named_parameters():
                if param_value.numel() == 1:
                    self.logger(f'params/{param_name}', param_value)
                else:
                    self.logger(f'params/{param_name}_mean', param_value.mean())
                    self.logger(f'params/{param_name}_std', param_value.std())

            #log plots/images more sparsely to save space
            if self.logger.iteration % (self.sparse_log_factor*self.print_every) == 0:
                fig = plots.plot_weights(self.net)[0].get_figure()
                self.logger.add_figure('weights', fig)

                debug_batch = data.get_aa_debug_batch(self.train_loader.dataset, n_per_class=10)
                debug_input, debug_target, debug_clamp_mask = self._prep_data(*debug_batch)
                state_debug_history = self.net(debug_input, clamp_mask=debug_clamp_mask, debug=True)
                fig = plots.plot_hidden_max_argmax(state_debug_history, n_per_class=None)[0].get_figure()
                self.logger.add_figure('hidden', fig)

            self.net.train()
            print(log_str)



class SGDTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, **kwargs):
        #explicit test_loader kwarg also allows it to be passed as positional arg
        super().__init__(net, train_loader, test_loader, **kwargs)
        self.name = 'SGD'


    def _update_parameters(self, output, target):
        loss = self.compute_loss(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss



class FPTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, **kwargs):
        #explicit test_loader kwarg also allows it to be passed as positional arg
        super().__init__(net, train_loader, test_loader, **kwargs)
        self.name = 'FPT'
        assert type(self.loss_fn) == nn.MSELoss, 'dLdW only implemented for MSE loss'
        self.verify_grad = False #for sanity-checking dLdW against non-batched version


    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


    def _update_parameters(self, output, target):
        #TODO: https://pytorch.org/docs/stable/notes/extending.html
        self.compute_gradients(output, target)
        self.optimizer.step()


    def compute_gradients(self, output, target):
        dLdW = self.compute_dLdW(output, target)
        if self.net.normalize_weight:
            #W is normalized, _W is not, need one more step of chain rule
            Z = self.net._W.norm(dim=1, keepdim=True) #[N,M]->[N,1]
            S = (dLdW * self.net._W).sum(dim=1, keepdim=True) #[N,M],[N,M]->[N,1]
            self.net._W.grad = dLdW / Z - self.net._W * (S / Z**3)
        else:
            self.net._W.grad = dLdW

        if self.reg_loss is not None:
            self.net._W.grad -= self.reg_loss.grad()['_W']

        if self.net.beta.requires_grad:
            self.net.beta.grad = self.compute_dLdB(output, target)

            # dLdW_nb = self.compute_dLdW_nobatch(output.squeeze(0), target.squeeze(0))
            # assert (dLdW-dLdW_nb).abs().mean() < 1e-10, 'Error in dLdW calculation!'

        self.write_gradients_log(self.net._W.grad, dLdW, self.net.beta.grad)


    def write_gradients_log(self, dLd_W=None, dLdW=None, dLdB=None):
        if self.logger.iteration % self.print_every == 0:
            self.logger('train/grad_norm', self.net._W.grad.norm())
            if dLdB is not None:
                self.logger('train/dLdB', dLdB)

        if self.logger.iteration % (self.sparse_log_factor*self.print_every) == 0:
            if self.net.normalize_weight:
                axs = plots.plot_weights(_W=dLd_W, W=dLdW)
                axs[0].set_title('dL/d_W (gradient step)')
                axs[1].set_title('dL/dW (wrt. normalized)')
                fig = axs[0].get_figure()
            else:
                fig = plots.plot_weights(_W=dLd_W).get_figure()
            self.logger.add_figure('gradient', fig)


    def compute_dLdW(self, v, t):
        """
        Assumes:
            Modern Hopfield dynamics, tau*dv/dt = -v + W*softmax(W^T*v)
            v is a fixed point, dv/dt = 0, i.e. v = W*softmax(W^T*v)
            MSE loss, L = 1/2B sum_i sum_b (v_i - t_i)^2, where B is batch size
        """
        N,M = self.net.W.shape
        g = self.net._g(v)  # [B,M,1]
        h = self.net._h(g)  # [N,M]@[B,M,1] -> [B,N,1]
        f = self.net._f(h)  # [B,N,1]
        del h

        if self.loss_mode == 'class':
            e = v[:, -self.n_class_units:] - t[:, -self.n_class_units:]  # error [B,Mc,1]
        else:
            e = v-t #[B,M,1]
        del v, t  # free up memory

        # ([B,N,N]-[B,N,1]@[B,1,N])@[N,M] -> [B,N,M]
        bFFW = self.net.beta * (f.squeeze().diag_embed() - f @ f.transpose(1, 2)) @ self.net.W
        A = torch.eye(M, device=self.device) - self.net.W.t() @ bFFW  # [M,M]+[M,N]@[B,N,M] -> [B,M,M]

        if self.loss_mode == 'class': # only compute loss over the last Mc units, M=Md+Mc
            Ainv = torch.linalg.inv(A)  # [B,M,M]
            a = Ainv[:, -self.n_class_units:].transpose(1, 2) @ e #[B,M,Mc]@[B,Mc,1] -> [B,M,1]
            del Ainv
        else:
            # a = A^(-1)^T * e <=> a = A^(-1) * e  b/c A and A^(-1) symmetric for g(x)=x
            a = torch.linalg.solve(A.transpose(1, 2), e)  # [B,M,M],[B,M,1] -> [B,M,1]
        del A, e

        #[B,N,1]@[B,1,Mc]+[B,N,M]@[B,M,1]@[B,1,M] -> [B,N,M]
        dLdW = f @ a.transpose(1,2) + bFFW @ a @ g.transpose(1,2)
        del f, a, bFFW, g
        return dLdW.mean(dim=0)


    def compute_dLdB(self, v, t):
        g = self.net._g(v)  # [B,M,1]
        h = self.net._h(g)  # [N,M]@[B,M,1] -> [B,N,1]
        f = self.net._f(h)  # [B,N,1]

        dfdB = f * (h - h.transpose(1,2)@f) #[B,N,1]
        dvdB = self.net.W.t() @ dfdB #[M,N]@[B,N,1] -> [B,M,1]
        dLdB = (v-t).transpose(1,2) @ dvdB #[B,1,1]
        return dLdB.mean(0).squeeze()


    def compute_dLdW_nobatch(self, v, t): #for debugging
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


    def numerical_dLdW(self, input, target, eps=1e-6):
        _W_flat = self.net._W.flatten() #reference
        dLd_W = torch.empty_like(_W_flat)
        for i,w in enumerate(_W_flat):
            _W_flat[i] = w + eps
            output = self.net(input)
            loss_plus = self.compute_loss(output, target)
            _W_flat[i] = w

            _W_flat[i] = w - eps
            output = self.net(input)
            loss_minus = self.compute_loss(output, target)
            _W_flat[i] = w

            dLd_W[i] = loss_plus - loss_minus
        dLd_W = dLd_W.reshape(self.net._W.shape)/(2*eps)
        return dLd_W


    def numerical_dLdB(self, input, target, eps=1e-6):
        beta = deepcopy(self.net.beta)

        self.net.beta = beta + eps
        output = self.net(input)
        loss_plus = self.compute_loss(output, target)
        self.net.beta = beta

        self.net.beta = beta - eps
        output = self.net(input)
        loss_minus = self.compute_loss(output, target)
        self.net.beta = beta

        dLdB = (loss_plus - loss_minus)/(2*eps)
        return dLdB


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
        details_str = f' ({self.name})' if self.name is not None else ''
        elapsed_str = 'Time elapsed{}: {} sec'.format(details_str, elapsed)
        print(elapsed_str)


class CustomOpt():
    #TODO: inherit from torch.optim.Optimizer
    def __init__(self, net, lr=1., lr_decay=1., momentum=False, rescale_grad=False,
                 clip=False, clip_thres=None, beta_increment=False, beta_max=None, **kwargs):
        if kwargs:
            #kwargs ignores any extra values passed in (eg. via a Config object)
            warnings.warn(f'Ignoring unused FPTOptimizer kwargs: {kwargs}')

        self.net = net

        self.lr = lr
        self.lr_decay = lr_decay

        #https://distill.pub/2017/momentum/
        self.momentum = momentum
        self.grad_avg = 0

        #https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
        assert clip in ['norm', 'value', False]
        self.clip = clip
        self.clip_thres = clip_thres

        self.rescale_grad = rescale_grad

        self.beta_increment = beta_increment
        self.beta_max = beta_max


    def step(self):
        #update _W
        dLd_W = self.net._W.grad

        if self.momentum:
            self.grad_avg = dLd_W + self.momentum*self.grad_avg
            dLd_W = self.grad_avg

        if self.clip == 'norm':
            norm = dLd_W.norm()
            if norm > self.clip_thres:
                dLd_W = self.clip_thres*dLd_W/norm
        elif self.clip == 'value':
            dLd_W[dLd_W > self.clip_thres] = self.clip_thres
            dLd_W[dLd_W < -self.clip_thres] = -self.clip_thres

        if self.rescale_grad:
            dLd_W = dLd_W/dLd_W.max(dim=1, keepdim=True)[0]

        self.net._W -= self.lr * dLd_W

        #update beta
        if self.beta_increment and self.net.beta < self.beta_max:
            self.net.beta += self.beta_increment

        if self.net.beta.requires_grad:
            self.net.beta -= self.lr * self.net.beta.grad

        #learning rate
        if self.lr_decay:
            self.lr *= self.lr_decay


class Logger(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_acc = EventAccumulator(self.get_logdir())

        self.epoch = 0 #number of times through the entire dataset
        self.iteration = 0 #number of batches, depends on batch size

        self._new_style = False


    def __getitem__(self, key):
        # https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically
        self.event_acc.Reload() #TODO: necessary/expensive every time? can add "is_stale" flag
        try:
            walltimes, steps, vals = zip(*self.event_acc.Scalars(key))
        except: #if used "new style" in SummaryWriter.add_scalar()
           walltimes, steps, vals = zip(*self.event_acc.Tensors(key))
           vals = [v.float_val[0] for v in vals]
        return steps, vals


    def __call__(self, *args, **kwargs):
        self.add_scalar(*args, **kwargs)


    def __contains__(self, key):
        for tag, keylist in self.event_acc.Tags().items():
            try:
                if key in keylist: return True
            except:
                pass #not every keylist is actually a list, ignore it if it's not
        return False


    def plot_key(self, key, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(*self[key], **kwargs)
        return ax


    def to_dict(self):
        """Returns a dictionary in the format {key : (steps_list values_list)} for every scalar key
        stored by the logger. Excludes images, histograms, etc."""
        log_dict = {}
        scalar_keys = self.event_acc.Tags()['scalars'] + self.event_acc.Tags()['tensors']
        for key in scalar_keys:
            log_dict[key] = self[key]
        return log_dict


    def add_scalar(self, tag, scalar_value, global_step=None): #overrides SummaryWriter method
        try: scalar_value = scalar_value.item()
        except AttributeError: pass
        step = self.iteration if global_step is None else global_step
        super().add_scalar(tag, scalar_value, global_step=step, walltime=None, new_style=self._new_style)


    def add_figure(self, tag, scalar_value, global_step=None, **kwargs): #overrides SummaryWriter method
        step = self.iteration if global_step is None else global_step
        super().add_figure(tag, scalar_value, global_step=step, **kwargs)



class L2Reg(nn.Module):
    def __init__(self, params, reg_rate=1, scale_by_size=False):
        """Params is a dict e.g. dict(net.named_parameters()) or subset of named_parameters()"""
        assert type(params) == dict
        self.reg_rate = reg_rate
        if scale_by_size:
            self.num_params = sum([param.numel() for param in params.values()])
            self.reg_rate = self.reg_rate/self.num_params
        self.params = params

    def __call__(self):
        params_norm = sum([param.norm() for param in self.params.values()])
        reg_loss = self.reg_rate/2 * params_norm
        return reg_loss

    def grad(self): #only used for FPTrain where I'm computing gradient manually
        return {name: self.reg_rate*param for name, param in self.params.items()}


def classifier_acc(output, target):
    output_class = torch.argmax(output, dim=1)
    target_class = torch.argmax(target, dim=1)
    return l0_acc(output_class, target_class)


def l0_acc(output, target):
    return (output == target).float().mean()


def l1_acc(output, target):
    return 1 - F.l1_loss(output, target)


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
