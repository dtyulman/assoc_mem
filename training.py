#built-in
import warnings, time, gc
from copy import deepcopy

#third party
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#custom
import plots, data, networks
from networks import eye_like

class AssociativeTrain():
    def __init__(self, net, train_loader, test_loader=None, optimizer=None,
                 loss_mode='class', loss_fn='mse', acc_mode='class', acc_fn='L0', reg_loss=None,
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

                    input, target = self._prep_data(*batch)
                    del batch
                    #gc.collect()
                    torch.cuda.empty_cache()

                    output = self.net(input)
                    if torch.isnan(output).any():
                        raise RuntimeError('NaNs in output of network')
                    self._update_parameters(output, target, input)
                    del input
                    #bug/feature: logs the resulting loss/acc *after* the param update,
                    #not the loss which caused the update
                    self.write_log(output=output, target=target)

                    del output, target
                    #gc.collect()
                    torch.cuda.empty_cache()

        self.net.eval() #put net back in 'evaluation' mode e.g. disable dropout
        return self.logger


    def _prep_data(self, input, target, perturb_mask):
        input = input.to(self.device)
        target = target.to(self.device)
        clamp_mask = ~perturb_mask if ('clamp' in self.net.input_mode) else None
        if clamp_mask is not None:
            clamp_mask = clamp_mask.to(self.device)
        return (input, clamp_mask), target


    def _update_parameters(self, output, target, input=None):
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
    def write_log(self, input=None, target=None, output=None, train_loss=None, train_acc=None):
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
                        # batch = next(iter(self.train_loader))
                        batch = self.train_loader.dataset[:self.train_loader.batch_size]
                        input, target = self._prep_data(*batch)
                    output = self.net(input)

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
                # test_batch = next(iter(self.test_loader))
                test_batch = self.test_loader.dataset[:self.test_loader.batch_size]
                test_input, test_target = self._prep_data(*test_batch)

                test_output = self.net(test_input)
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
            # if self.logger.iteration % (self.sparse_log_factor*self.print_every) == 0:
            #     ax = plots.plot_weights(self.net)
            #     try:
            #         fig = ax[0].get_figure()
            #     except:
            #         fig = ax.get_figure()
            #     self.logger.add_figure('weights', fig)

                # #note, this grabs the first 10 of each class, not the first 100 samples, so if
                # #initializing weights with data, it's not guaranteed that the first 100 hidden units
                # #will be most active for the debug batch
                # debug_batch = data.get_aa_debug_batch(self.train_loader.dataset, n_per_class=10)
                # debug_input, debug_target = self._prep_data(*debug_batch)
                # state_debug_history = self.net(debug_input, debug=True)
                # fig = plots.plot_hidden_max_argmax(state_debug_history, n_per_class=None)[0].get_figure()
                # self.logger.add_figure('hidden', fig)

            self.net.train()
            print(log_str)



class BPTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, **kwargs):
        #explicit test_loader kwarg also allows it to be passed as positional arg
        super().__init__(net, train_loader, test_loader, **kwargs)
        self.name = 'SGD'


    def _update_parameters(self, output, target, input=None):
        loss = self.compute_loss(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss



class FPTrain(AssociativeTrain):
    def __init__(self, net, train_loader, test_loader=None, verify_grad=False,
                 approx=False, **kwargs):
        #explicit test_loader kwarg also allows it to be passed as positional arg
        super().__init__(net, train_loader, test_loader, **kwargs)
        self.name = 'FPT'
        assert type(self.loss_fn) == nn.MSELoss, 'dLdW only implemented for MSE loss'
        self.verify_grad = verify_grad
        assert approx in [False, 'first', 'first-heuristic', 'inv-heuristic']
        self.approx = approx


    def __call__(self, *args, **kwargs):
        if self.verify_grad:
            return super().__call__(*args, **kwargs)
        else:
            with torch.no_grad():
                return super().__call__(*args, **kwargs)


    def _update_parameters(self, output, target, input=None):
        #TODO: https://pytorch.org/docs/stable/notes/extending.html
        self.store_gradients(output, target, input=input)
        self.optimizer.step()
        #gc.collect()
        torch.cuda.empty_cache() #should not be necessary but seems to help with OOM


    def store_gradients(self, output, target, input=None):
        #compute FPT grad
        unclamped_mask = None
        if 'clamp' in self.net.input_mode:
            #input shape is ([B,M,1], [M,1])
            assert type(input) == tuple
            assert len(input) == 2
            B = output.shape[0]
            for i in range(B-1):
                assert (input[1][i]==input[1][i+1]).all(), 'clamp mask must be the same for entire batch'
            unclamped_mask = ~input[1][0].squeeze() #[M,1]->[M]
        dLdW, dLdB = self.compute_gradients(output, target, unclamped_mask)

        dLd_W = dLdW
        if self.net.normalize_weight:
            #net.W is normalized, net._W is not, need one more step of chain rule
            Z = self.net._W.norm(dim=1, keepdim=True) #[N,M]->[N,1]
            S = (dLdW * self.net._W).sum(dim=1, keepdim=True) #[N,M],[N,M]->[N,1]
            dLd_W = dLdW / Z - self.net._W * (S / Z**3)

        if self.reg_loss is not None:
            dLd_W += self.reg_loss.grad()['_W']

        self.verify_grad_fn(dLd_W, dLdB, input, output, target)

        #save FPT grads (overwrites BPTT grads if those were stored during verify_grad)
        self.net._W.grad = dLd_W
        if type(self.net.f) == networks.Softmax and self.net.f.beta.requires_grad:
            self.net.f.beta.grad = dLdB

        if type(self.net.f) == networks.Softmax:
            self.write_gradients_log(self.net._W.grad, dLdW, self.net.f.beta.grad)
        else:
            self.write_gradients_log(self.net._W.grad, dLdW)


    def verify_grad_fn(self, dLd_W, dLdB, input, output, target):
        #compare FPT w/ BPTT and numerical
        if 'bptt' in self.verify_grad:
            print('Checking gradients (BPTT)...')

            #store BPTT grad to compare w/ FPT
            self.optimizer.zero_grad()
            loss = self.compute_loss(output, target)
            loss.backward()

            _,ax = plt.subplots(3,4)
            [a.axis('off') for a in ax.flatten()]
            plots._plot_rows(input[0], title='in', ax=ax[0,0])
            plots._plot_rows(output, title='out', ax=ax[0,1])
            plots._plot_rows(target, title='tgt', ax=ax[0,2])
            plots._plot_rows(output-target, title='err = out-tgt', ax=ax[0,3])

            plots._plot_rows(dLd_W, title='fpt', ax=ax[1,0])
            plots._plot_rows(self.net._W.grad, title='bptt', ax=ax[1,1])
            plots._plot_rows(dLd_W-self.net._W.grad, title='fpt-bptt', ax=ax[2,0])
            check_close(self.net._W.grad, dLd_W, 'dLd_W_BPTT', 'dLd_W_FPT')
            if self.net.f.beta.requires_grad:
                check_close(self.net.f.beta.grad, dLdB, 'dLdB_BPTT', 'dLdB_FPT')

        if 'num' in self.verify_grad:
            print('Checking gradients (numerical)...')
            dLd_W_num = self.numerical_dLd_W(input, target)
            plots._plot_rows(dLd_W_num, title='num', ax=ax[1,2])
            plots._plot_rows(dLd_W-dLd_W_num, title='fpt-num', ax=ax[2,1])
            check_close(dLd_W_num, dLd_W, 'dLd_W_NUM', 'dLd_W_FPT')
            if self.net.f.beta.requires_grad:
                dLdB_num = self.numerical_dLdB(input, target)
                check_close(dLdB_num, dLdB, 'dLdB_NUM', 'dLdB_FPT')

        if 'num' in self.verify_grad and 'bptt' in self.verify_grad:
            plots._plot_rows(self.net._W.grad-dLd_W_num, title='bptt-num', ax=ax[2,2])
            check_close(dLd_W_num, self.net._W.grad, 'dLd_W_NUM', 'dLd_W_BPTT')
            if self.net.f.beta.requires_grad:
                check_close(dLdB_num, self.net.f.beta.grad, 'dLdB_NUM', 'dLdB_BPTT')



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


    def compute_gradients(self, g, g_target, unclamped_mask=None):
        """
        g, g_target are [B,M,1]
        unclamped_mask is [M] (same mask for entire batch)
        """
        B,M,_ = g.shape

        gradL = g - g_target; del g_target #[B,M,1], assumes MSE loss
        if self.loss_mode == 'class':
            #TODO: generalize to arbitrary readout_mask
            gradL[:, :-self.n_class_units] = 0 #only compute loss over the last Mc units, M=Md+Mc
            gradL = 2/self.n_class_units*gradL #correct for loss normalization to match nn.MSELoss
        else:
            gradL = 2/M*gradL #correct for loss normalization to match nn.MSELoss ((v-t)^2).mean()

        W = self.net.W
        Jg = self.net.g.J(self.net.v_last, f=g)
        if 'clamp' in self.net.input_mode:
            #TODO: generalize to different mask per item in batch
            #(put in zeros to match dims or force mask dims per batch)
            W = W[:, unclamped_mask] #[N,Mu]
            Jg = Jg[:,:,unclamped_mask] #[B,M,Mu]


        h = self.net.compute_hidden(g)
        if self.approx == 'first': #assumes f(h) = softmax(beta*h)
            f_first_order = networks.Softmax_1()
            f = f_first_order._zeroth_order(h) #not really f, but plays its role when calculating dLdW
            Jf = f_first_order.J(h, f0=f)
            D_beta = f_first_order.D_beta(h)
        else:
            f = self.net.f(h)
            Jf = self.net.f.J(None, f)
            D_beta = self.net.f.D_beta(h)
        del h

        if self.approx == False:
            A = eye_like(gradL) - Jg @ W.t() @ Jf @ self.net.W #[M,M]+[B,M,Mu]@[Mu,N]@[B,N,N]@[N,M]->#[B,M,M]
            #a = Jg^T * A^T^(-1) * gradL
            a = a_ = Jg.transpose(-2,-1) @ torch.linalg.solve(A.transpose(-2,-1), gradL); del A #[B,M,1]
        elif self.approx == 'inv-heuristic':
            a = a_ = gradL
        elif self.approx == 'first' or self.approx == 'first-heuristic':
            Ainv = eye_like(gradL) + Jg @ W.t() @ Jf @ self.net.W
            a  = Jg.transpose(-2,-1) @ Ainv.transpose(-2,-1) @ gradL; del Ainv
            a_ = Jg.transpose(-2,-1) @ gradL
        else:
            raise ValueError()
        del Jg

        if 'clamp' in self.net.input_mode:
            dLdW = torch.zeros(B, *self.net.W.shape, device=a.device)
            dLdW[:,:,unclamped_mask] = f @ a.transpose(-2,-1)
        else:
            dLdW = f @ a.transpose(-2,-1)
        dLdW += Jf.transpose(-2,-1) @ W @ a_ @ g.transpose(-2,-1)
        dLdW = dLdW.mean(0)

        dLdB = ((W @ a_).transpose(-2,-1) @ D_beta).mean()

        return dLdW, dLdB


    def numerical_dLd_W(self, input, target):
        self.net._W.requires_grad_(False)
        _W_flat = self.net._W.flatten() #view of _W, so modifying _W_flat also modifies _W
        dLd_W = torch.empty_like(_W_flat)

        eps = torch.tensor(torch.finfo(torch.get_default_dtype()).eps).item()**(1./3)
        for i,w in enumerate(_W_flat):
            #optimal step size: http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-7.pdf
            h_i = _W_flat[i].item()*eps

            w_orig = deepcopy(w.item())

            _W_flat[i] += h_i
            output_plus = self.net(input)
            loss_plus = self.compute_loss(output_plus, target)
            _W_flat[i] = w_orig

            _W_flat[i] -= h_i
            output_minus = self.net(input)
            loss_minus = self.compute_loss(output_minus, target)
            _W_flat[i] = w_orig

            dLd_W[i] = (loss_plus - loss_minus)/(2*h_i)
        dLd_W = dLd_W.reshape(self.net._W.shape)

        self.net._W.requires_grad_(True)
        return dLd_W


    def numerical_dLdB(self, input, target, eps='auto'):
        self.net.f.beta.requires_grad_(False)
        eps = torch.tensor(torch.finfo(torch.get_default_dtype()).eps)**(1./3)
        h = self.net.f.beta.item()*eps

        beta_orig = deepcopy(self.net.f.beta.data)

        self.net.f.beta += h
        output_plus = self.net(input)
        loss_plus = self.compute_loss(output_plus, target)
        self.net.f.beta.data = beta_orig

        self.net.f.beta -= h
        output_minus = self.net(input)
        loss_minus = self.compute_loss(output_minus, target)
        self.net.f.beta.data = beta_orig

        dLdB = (loss_plus - loss_minus)/(2*h)

        self.net.f.beta.requires_grad_(True)
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
        if self.beta_increment and self.net.f.beta < self.beta_max:
            self.net.f.beta += self.beta_increment

        if self.net.f.beta.requires_grad:
            self.net.f.beta -= self.lr * self.net.f.beta.grad

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


    def add_figure(self, tag, figure, global_step=None, **kwargs): #overrides SummaryWriter method
        step = self.iteration if global_step is None else global_step
        super().add_figure(tag, figure, global_step=step, **kwargs)



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


########
# Misc #
########
def check_close(tensor1, tensor2, str1=None, str2=None, mode='warn', **kwargs):
    if str1 is None:
        str1 = 'Tensor 1'
    if str2 is None:
        str2 = 'Tensor 2'

    max_abs_diff = (tensor1-tensor2).abs().max()
    debug_msg = f'{str1} {{}} {str2}, MaxAbsDiff={max_abs_diff}'

    if torch.allclose(tensor1, tensor2, **kwargs):
        print(debug_msg.format('same as'))
    else:
        debug_msg = debug_msg.format('!=')
        if mode == 'raise':
            raise Exception(debug_msg)
        elif mode == 'warn':
            print('!--> ' + debug_msg)
        else:
            raise ValueError(f"Invalid check mode: '{mode}'")


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
