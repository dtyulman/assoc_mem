import math, types
from copy import deepcopy

import torch
import torchvision
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import plots
import callbacks as cb
import components as nc


class AssociativeMemory(pl.LightningModule):
    def __init__(self, dt=1., converged_thres=0, max_steps=100, warn_not_converged=True,
                 input_mode = 'init', #init, clamp
                 train_kwargs={'mode':'bptt'},
                 optimizer_kwargs={'class':'Adam'},
                 sparse_log_factor=5):
        super().__init__()
        self.save_hyperparameters(ignore=['sparse_log_factor', 'warn_not_converged'])

        self.input_mode = input_mode

        self.dt = dt
        self.max_steps = max_steps
        self.converged_thres = converged_thres
        self.warn_not_converged = warn_not_converged

        self.sparse_log_factor = sparse_log_factor
        self.setup_training_step(**train_kwargs)

        #TODO: specify visible=True/False in the Layer object and auto register it to this list
        self.visible_layers = [0] #by default assume the 0th entry of state tuple is the input/output layer
        self.state_shape = None #excludes batch dimension, gets set by self.pack()


    def forward(self, *init_state, clamp_mask=None, return_mode=''):
        """
        init_state: list of initial states for each layer in network. Set list entry
            to None to use the Layer default. At least one entry must be provided.
        clamp_mask: same shape as init_state[0], selects which entries of the state to clamp
        return_mode: list of options to return, can include all/any
            'trajectory': returns entire trajectory of the state evolution from init to fixed-point
            'converge_time': returns iteration at which each item in the batch converged
        """
        state = self.default_init(*init_state)
        if 'clamp' in self.input_mode and clamp_mask is not None:
            #TODO: this assumes state[0] is the (perturbed) "input". In general might be
            # any/some/all of the entries in the state tuple
            clamp_values = state[0][clamp_mask]

        if 'trajectory' in return_mode:
            state_trajectory = [state]
        if 'converge_time' in return_mode:
            converge_time = torch.ones(state[0].shape[0])*self.max_steps+1
        for t in range(1,self.max_steps+1):
            prev_state = state
            state = self.step(*prev_state)
            if 'clamp' in self.input_mode and clamp_mask is not None:
                state[0][clamp_mask] = clamp_values #TODO: same as above

            if 'trajectory' in return_mode:
                state_trajectory.append(state)
            if 'converge_time' in return_mode:
                residual_per_item = self.get_residual(state, prev_state, batch_avg=False)
                converge_time[(converge_time==self.max_steps+1) & (residual_per_item<=self.converged_thres)] = t

            if 'converge_time' in return_mode:
                residual = residual_per_item.max()
            else:
                residual = self.get_residual(state, prev_state)
            if residual <= self.converged_thres:
                break

        self.log('steps_to_converge', t)
        if residual > self.converged_thres and self.warn_not_converged:
            print(f'FP not converged: t={t}, residual={residual:.1e}, thres={self.converged_thres}')

        ret = state_trajectory if 'trajectory' in return_mode else state
        ret = (ret, converge_time) if 'converge_time' in return_mode else ret
        return ret


    def pack(self, state, layers='all'):
        """Returns a flattened version the state tuple, e.g. ([B,M],[B,N])->[B,M+N].
        Specify layers as list of indices to select which layers to pack or 'visible' to pack all
            visible layers. e.g. layers=[0] or layers='visible' gives ([B,M],[B,N])->[B,M]
        """
        self.state_shape = [s.shape[1:] for s in state]
        if layers == 'all':
            layers = range(len(state))
        elif layers == 'visible':
            layers = self.visible_layers
        return torch.cat([state[i].flatten(start_dim=1) for i in layers], dim=1) #[B,V]


    def unpack(self, packed_state):
        """Undoes the pack(state, layers='all') operation. Only works if the entire state was packed."""
        batch_size = packed_state.shape[0]
        i = 0
        unpacked = []
        for shape in self.state_shape:
            j = i+math.prod(shape)
            unpacked.append(packed_state[:,i:j].reshape(batch_size, *shape))
            i = j
        return unpacked


    def get_residual(self, state, prev_state, batch_avg=True):
        """ Computes the relative residual over the visible neurons
        state, prev_state: tuple of tensors where i-th entry is tensor of activations from layer i
            eg. ([B,M],[B,N]) for LargeAssocMem or ([B,Cx,L,L],[B,Cy,M,M],[B,N]) for ConvThreeLayer
        batch_avg: if True computes scalar residual over entire batch. Otherwise, returns length-B
            tensor with residuals for each element in the batch
        """
        with torch.no_grad():
            state = self.pack(state, layers='visible')
            prev_state = self.pack(prev_state, layers='visible')
            if batch_avg:
                return ((state - prev_state).norm()/state.norm()).item()
            return (state - prev_state).norm(dim=1)/state.norm(dim=1)


    def default_init(self, *init_state):
        """Override if relying on default initialization for any layers. Returns
        full initial state (tuple of tensors, same dims as the state tuple)"""
        return init_state


    def step(self, *prev_state):
        """Override. Returns next state (tuple of tensors)"""
        raise NotImplementedError()


    def energy(self, *state):
        """Override. Returns energy function (scalar) evaluated at given state"""
        raise NotImplementedError()


    @staticmethod
    def infer_input_size(train_data):
        """Override. Returns dict of input dimensions for network"""
        raise NotImplementedError()


    def configure_optimizers(self):
        optimizer_kwargs = deepcopy(self.hparams.optimizer_kwargs)
        Optimizer = getattr(torch.optim, optimizer_kwargs.pop('class'))
        return Optimizer(self.parameters(), **optimizer_kwargs)


    def configure_callbacks(self):
        return [cb.ParamsLogger(), cb.OutputsLogger(self.sparse_log_factor)]


    @staticmethod
    def loss_fn(output, target):
        numel = float(sum(o.numel() for o in output))
        total_loss = sum(F.mse_loss(o,t, reduction='sum') for o,t in zip(output,target))
        return total_loss/numel


    @staticmethod
    def acc_fn(output, target):
        numel = float(sum(o.numel() for o in output))
        total_err = sum(torch.dist(o,t,0) for o,t in zip(output,target))
        return 1-total_err/numel


    def training_step(self, batch, batch_idx):
        input, target, perturb_mask = batch[0:3]
        if isinstance(target, torch.Tensor):
            target = (target,) #output is tuple, make sure target is too
        output = self.training_forward(input, clamp_mask=~perturb_mask)
        loss = self.loss_fn(output, target)
        acc = self.acc_fn(output, target)

        self.log('train/loss', loss.item())
        self.log('train/acc', acc.item())

        return {'loss': loss,
                'output': tuple(o.detach() for o in output)}


    def setup_training_step(self, mode, **kwargs):
        self.train_mode = mode
        if mode == 'bptt':
            def training_forward(self, input, clamp_mask=None, return_mode=''):
                return self(input, clamp_mask=clamp_mask, return_mode=return_mode)

        elif mode == 'rbp':
            self.rbp_max_steps = kwargs.pop('max_steps', 5000)
            self.rbp_converged_thres = kwargs.pop('converged_thres', 1e-6)

            def training_forward(self, input, clamp_mask=None):
                #basic idea from here: https://implicit-layers-tutorial.org/deep_equilibrium_models/
                with torch.no_grad():
                    fp = self(input, clamp_mask=clamp_mask)
                fp = self.pack(self.step(*fp)) #packing so that autograd gets engaged properly

                fp_in = fp.detach().clone().requires_grad_()
                fp_out,_ = self.pack(self.step(*self.unpack(fp_in)))
                def backward_hook(grad):
                    new_grad = grad
                    for t in range(1,self.rbp_max_steps+1):
                        new_grad_prev = new_grad
                        new_grad = torch.autograd.grad(fp_out, fp_in, new_grad_prev, retain_graph=True)[0] + grad
                        residual = ((new_grad - new_grad_prev).norm()/new_grad.norm()).item()
                        if residual <= self.rbp_converged_thres:
                            break
                    if t > self.rbp_max_steps:
                        print(f'BP not converged: t={t}, res={residual}')
                    return new_grad
                fp.register_hook(backward_hook)
                return self.unpack(fp)

        elif self.train_mode == 'rbp-1h':
            def training_forward(input, clamp_mask=None):
                pass
        else:
            raise ValueError('Invalid train_mode: {self.train_mode}')

        self.training_forward = types.MethodType(training_forward, self)


    def get_energy_trajectory(self, batch, debug=False):
        input, target, perturb_mask = batch[0:3]
        with torch.no_grad():
            state_trajectory = self(input, clamp_mask=~perturb_mask, return_mode='trajectory')
        #[T x ([B,*] ... [B,*])] -> ([T,B,*], ..., [T,B,*])
        #TODO: store immediately in this format to avoid reshaping? Would need to rewrite other methods
        state_trajectory = [torch.stack(layer_tr) for layer_tr in zip(*state_trajectory)]
        return self.energy(*state_trajectory, debug=debug)


    def plot_energy_trajectory(self, batch):
        with torch.no_grad():
            energy_trajectory = self.get_energy_trajectory(batch) #([T,B,*], ..., [T,B,*])

        fig, ax = plt.subplots()
        ax.plot(energy_trajectory)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')

        return fig, ax



class LargeAssociativeMemory(AssociativeMemory):
    """From Krotov and Hopfield 2020
    g: [B,N] (feature layer firing rate)
    f: [B,M] (hidden layer firing rate)
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 input_nonlin = nc.Identity(),
                 hidden_nonlin = nc.Softmax(),
                 rescale_grads = False,
                 normalize_weights = False, #False, frobenius (or 'F' or True), rows, rows_scaled
                 tau = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=['input_nonlin', 'hidden_nonlin']) #nonlins already saved during checkpoiting b/c instances of nn.Module

        self.feature = nc.Layer(input_size, nonlin=input_nonlin, tau=float(tau), dt=self.dt)
        self.fc = nc.LinearNormalized(input_size, hidden_size, normalize_weights=normalize_weights)
        self.hidden = nc.Layer(hidden_size, nonlin=hidden_nonlin, tau=0, dt=self.dt)

        self.rescale_grads = rescale_grads


    def default_init(self, input_init, hidden_init=None):
        if hidden_init is None:
            hidden_init = self.hidden.default_init(input_init.shape[0]).to(input_init.device)
        return input_init, hidden_init


    def energy(self, g, f, debug=False):
        """
        g: [T,B,N] or [B,N] or [N]
        f: [T,B,M] or [B,M] or [M]

        Change of variables h=W*g and v=W^T*f (see documentation for `step`):

            E = [v   *G(v)    - L_v(v)]    + [h *F(h)  - L_h(h)]  - F(h) *W*G(v) #from KH2020
              = [W^Tf*G(W^Tf) - L_v(W^Tf)] + [Wg*F(Wg) - L_h(Wg)] - F(Wg)*W*G(W^Tf)

        where G(.) is feature.nonlin and F(.) is hidden.nonlin
        """
        #change of variables
        v = self.fc.T(f) #[T,B,N]
        h = self.fc(g) #[T,B,M]

        G = self.feature.nonlin(v) #[T,B,N]
        F = self.hidden.nonlin(h) #[T,B,M]

        # E_feature = self.feature.energy(v)
        Lv = torch.linalg.vector_norm(v, dim=-1) #[T,B]
        E_feature = (v*G).sum(dim=-1) - Lv #[T,B]

        # E_hidden = self.hidden.energy(h)
        beta = self.hidden.nonlin.beta
        Lh = torch.logsumexp(beta*h, dim=-1)/beta #[T,B]
        E_hidden = (h*F).sum(dim=-1) - Lh #[T,B]

        # E_syn = self.fc.energy(F, G)
        #[T,B,1,M]@[M,N]@[T,B,N,1]->[T,B,1,1]->[T,B]
        E_syn = -(F.unsqueeze(-2) @ self.fc.weight @ G.unsqueeze(-1)).squeeze()

        E = E_feature + E_hidden + E_syn #[T,B]
        if debug:
            return E_feature, E_hidden, E_syn, E
        return E


    def plot_energy_trajectory(self, batch, debug=False):
        if debug:
            with torch.no_grad(): #([T,B,*], ..., [T,B,*])
                energy_trajectory = self.get_energy_trajectory(batch, debug=debug)

            fig, ax = plt.subplots(2,2, sharex=True)
            _ax = ax.flatten()
            for i, label in enumerate(['$E_{feat}$', '$E_{hid}$', '$E_{syn}$', '$E_{tot}$']):
                _ax[i].plot(energy_trajectory[i])
                _ax[i].set_ylabel(label)
            [_ax[i].set_xlabel('Time') for i in [2,3]]
            fig.tight_layout()
            return fig, ax

        return super().plot_energy_trajectory(batch)


    def _compute_grad_scaling(self):
        """Must be called after backward()"""
        #_weight.shape is [hidden_size, input_size]
        return self.fc._weight.grad.abs().max(dim=1, keepdim=True)[0]


    def on_after_backward(self):
        """Optionally rescales gradient of each hidden unit's weights s.t. the max grad (across
        all afferent weights for that neuron) is equal to 1"""
        #TODO: can we move this to self.fc's class to make more general?
        if self.rescale_grads:
            self.fc._weight.grad /= self._compute_grad_scaling()


    def step(self, g_prev, f_prev=None):
        """
        dg/dt = -g + G(W^T*f)  <=>  g[t+dt] = (1-dt/tau) g[t] + dt/tau G( W^T*f[t] )
        df/dt = -f + F(W*g)    <=>  f[t+dt] = (1-dt/tau) f[t] + dt/tau F( W  *g[t] )

        Note that by change of variables h=W*g and v=W^T*f, this is equivalent to:
            dh/dt = -h + W*G(v)    <=>  h[t+dt] = (1-dt/tau) h[t] + dt/tau W  *G(v)
            dv/dt = -v + W^T*F(h)  <=>  v[t+dt] = (1-dt/tau) v[t] + dt/tau W^T*F(h)

            W*g[t+dt] = (1-dt/tau) W*g[t] + dt/tau W*G( W^T*f[t] )
            W^T*f[t+dt] = (1-dt/tau) W^T*f[t] + dt/tau W^T*F( W*g[t] )
        """
        #f_prev is ignored because tau_f=0 (included for function signature consistency)
        f = self.hidden.step(0, self.fc(g_prev))
        g = self.feature.step(g_prev, self.fc.T(f))
        return (g,f)


    @staticmethod
    def infer_input_size(train_data):
        """train_data: (X,Y) where X is [dataset_size, input_dim]"""
        return {'input_size' : train_data[0][0].numel()}


    def plot_weights(self, weights='weights', drop_last=0, pad_nan=True):
        if weights == 'weights':
            weights = self.fc.weight.detach()
            title = f'{self.__class__.__name__} weight'
        elif weights == 'weights_raw':
            weights = self.fc._weight.detach()
            title = f'{self.__class__.__name__} weight (raw)'
        elif weights == 'grads':
            weights = self.fc._weight.grad
            title = f'{self.__class__.__name__} grad_weight'
        elif not isinstance(weights, torch.Tensor):
            raise ValueError()

        imgs = plots.rows_to_images(weights, drop_last=drop_last, pad_nan=pad_nan)
        grid = plots.images_to_grid(imgs, vpad=1, hpad=1)
        fig, ax = plots.plot_matrix(grid, title=title)
        return fig, ax


    def on_before_optimizer_step(self, optimizer, opt_idx):
        """Use instead of cb.WeightsLogger() callback to plot unnormalized fc._weights in addition
        to fc.weights, and fc._weights.grad instead of fc.weights.grad"""
        # if (self.global_step+1) % (self.trainer.log_every_n_steps) == 0:
        #     if self.fc.normalize_mode == 'rows_scaled':
        #         self.trainer.logger.experiment.add_scalars(
        #             'row_scaling',
        #             {str(mu):alpha for mu,alpha in enumerate(self.fc.row_scaling)},
        #             global_step=self.global_step)
        if (self.global_step+1) % (self.trainer.log_every_n_steps*self.sparse_log_factor) == 0:
            with plots.NonInteractiveContext():
                fig = self.plot_weights(weights='weights')[0]
                self.trainer.logger.experiment.add_figure('params/weight', fig,
                                                     global_step=self.global_step)

                if self.fc.normalize_mode is False:
                    fig = self.plot_weights(weights='grads')[0]
                    self.trainer.logger.experiment.add_figure('grads/weight', fig,
                                                         global_step=self.global_step)
                else:
                    fig = self.plot_weights(weights='weights_raw')[0]
                    self.trainer.logger.experiment.add_figure('_params/_weight', fig,
                                                         global_step=self.global_step)

                    fig = self.plot_weights(weights='grads')[0]
                    self.trainer.logger.experiment.add_figure('_grads/_weight', fig,
                                                         global_step=self.global_step)




class ExceptionsMHN(LargeAssociativeMemory):
    def __init__(self, exceptions, beta=1, beta_exception=None, train_beta=True,
                 exception_loss_scaling=1, exception_loss_mode='manual',
                 *args, **kwargs):
        """
        exception_loss_scaling: defines the relative loss weighting for exceptions compared to
            non-exceptions. ALl loss weightings are normalized such that they sum to 1.
            Remember to run net.set_exception_loss_scaling() after initializing the network!
        exception_loss_mode:
            manual: selects exceptions based on known ground truth. Loss weighting for exceptions
                is `exception_loss_scaling` times the weighting for non-exceptions.
            entropy: entropy of hidden (softmax) layer. Items with max entropy (i.e. entropy of a
                discrete uniform distribution with M elements) get scaled proportional
                to `exception_loss_scaling`, zero entropy gets scaled proportional to `1` with
                linear interpolation in between
            max: Items whose max(hidden)==1 get scaled by `exception_loss_scaling`. Items with
                max(hidden)==1/M get scaled by `1`.
            norm: norm(hidden)==sqrt(M) gets scaled by `exception_loss_scaling`, norm==1 gets
                scaled by 1
            time: number of timesteps to reach the fixed point. time==net.max_steps gets scaled
                by 'exception_Loss_scaling', time==1 gets scaled by `1`

                #TODO: How to do better? This is bad because
                    1) max_steps is arbitrary and network might never hit this max
                    2) time==1 is unlikely, usually smallest number of of steps is ~200-300

            #TODO: possible issue: there's going to be significant interaction between beta and
                `exc_loss_scaling` because entropy range varies wildly (6.62-6.634 for beta=1)
                vs (0.5-6.5 for beta=10) in an trained network (all 0's, three 1's, M=100).
                Same for norm and max.
        """
        assert not train_beta or beta_exception is None, "Can't use dynamic beta if training beta0"
        kwargs.pop('hidden_nonlin', '') #TODO: hack to get this to load_from_checkpoint. Don't know why hidden_nonlin is in the kwargs in the first place....
        super().__init__(*args, hidden_nonlin=nc.Softmax(beta=beta, train=train_beta), **kwargs)

        self.beta_exception = torch.tensor(float(beta_exception)) if beta_exception is not None else None

        self.exceptions = set(exceptions)
        self.exception_loss_scaling = exception_loss_scaling
        self.exception_loss_mode = exception_loss_mode
        #remember to run net.set_exception_loss_scaling() after initializing the network!

        self.automatic_optimization = False #must be set *after* calling super().__init__()

        self.min_exception_signal = float('inf')
        self.max_exception_signal = -float('inf')


    def set_exception_loss_scaling(self, train_data):
        """Run this after initializing the network to set dataset-dependent variables!"""
        #TODO: write ExceptionsDataset which takes in a Dataset but keeps track of exceptions
        #and returns with __getitem__() a bool whether it's an exception for use with ExceptionsMHN
        #as optional ground truth. That way don't need to have this awkward method.

        is_exception = [l.item() in self.exceptions for l in train_data.labels]
        self.exceptions_idx = [idx for idx,is_ex in enumerate(is_exception) if is_ex]
        self.num_exceptions = len(self.exceptions_idx) #i.e. sum(is_exception)
        num_regulars = len(train_data)-self.num_exceptions

        if isinstance(self.exception_loss_scaling, (int,float)):
            pass
        elif self.exception_loss_scaling == 'linear_dataset':
            self.exception_loss_scaling = num_regulars/self.num_exceptions
        elif self.exception_loss_scaling == 'log_dataset':
            raise NotImplementedError
        else:
            raise ValueError()


    def configure_callbacks(self):
        """Don't use generic OutputsLogger because want to see exceptions in every sample batch"""
        callbacks = super().configure_callbacks()
        for c in callbacks:
            if isinstance(c, cb.OutputsLogger):
                del c
        return callbacks


    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Use instead of cb.OutputsLogger() to ensure every sample batch has exceptions in it"""
        if (self.global_step+1) % (self.trainer.log_every_n_steps*self.sparse_log_factor) == 0:
            input, target = batch[0:2]
            output = outputs['output'][0] #unpack tuple, assumes state[0] is the "output"

            #replace last `num_exceptions` elements in batch with the exceptions from the dataset
            input_ex, target_ex, perturb_mask_ex = \
                self.trainer.train_dataloader.dataset.datasets[self.exceptions_idx][0:3]
            with torch.no_grad():
                output_ex = self(input_ex, clamp_mask=~perturb_mask_ex)[0] #unpack tuple

            input[-self.num_exceptions:] = input_ex
            target[-self.num_exceptions:] = target_ex
            output[-self.num_exceptions:] = output_ex

            with plots.NonInteractiveContext():
                fig, ax = self.trainer.train_dataloader.dataset.datasets \
                            .plot_batch(inputs=input, targets=target, outputs=output)
            self.trainer.logger.experiment.add_figure('sample_batch', fig,
                                                 global_step=self.trainer.global_step)


    def on_after_backward(self):
        """Override parent class which rescales grads in this method. Make it a passthrough since
        we're rescaling grads inside training_step for additional flexibility wrt exceptions"""
        pass


    def training_step(self, batch, batch_idx):
        """Using manual optimization so that I can rescale gradients flexibly (other modules use
        automatic optim). Must set self.automatic_optimization=False in __init__()"""
        input, target, perturb_mask, label = batch
        if isinstance(target, torch.Tensor):
            target = (target,) #output is tuple, make sure target is too
        output, converge_time = self.training_forward(input, clamp_mask=~perturb_mask, return_mode='converge_time')
        hidden_activation = output[1]

        opt = self.optimizers()
        #compute gradients without weighting and store them for use in rescaling
        if self.rescale_grads == 'unweighted_loss':
            opt.zero_grad()
            unweighted_loss = self.loss_fn(output, target)
            self.manual_backward(unweighted_loss, retain_graph=True)
            unweighted_grad_scaling = self._compute_grad_scaling()

        #optionally re-run network with different beta for exceptions
        is_exception = torch.tensor([l.item() in self.exceptions for l in label])
        if self.beta_exception is not None and is_exception.any():
            with nc.HotSwapParameter(self.hidden.nonlin.beta, self.beta_exception):
                #remember to unpack length-1 output state tuple
                output[0][is_exception] = self.training_forward(
                    input[is_exception], clamp_mask=~perturb_mask[is_exception])[0]

        #manually compute gradients, optionally rescale, and step the optimizer
        #TODO: don't re-run this with 'unweighted_loss' grad scaling if there are no exceptions in
        #the batch, since unweighted_loss is same as loss (and the same as loss_reg)
        opt.zero_grad()
        loss, loss_reg, loss_exc = self.loss_fn_exceptions(output, target, is_exception,
                                                           hidden_activation, converge_time)
        self.manual_backward(loss)
        if self.rescale_grads == 'unweighted_loss':
            self.fc._weight.grad /= unweighted_grad_scaling
        elif self.rescale_grads is True:
            self.fc._weight.grad /= self._compute_grad_scaling()
        elif self.rescale_grads is not False:
            raise ValueError()
        opt.step()

        acc = self.acc_fn(output, target)

        self.log('train/loss', loss.item())
        if loss_reg is not None and loss_exc is not None:
            print(f'[exception] it={self.global_step}, ep={self.current_epoch}, bx={batch_idx}, '
                  f'loss={loss:.4e}, loss_exc={loss_exc:.4e}, loss_reg={loss_reg:.4e}')
            self.log('train/loss_exception', loss_exc.item())
            self.log('train/loss_regular', loss_reg.item())

        self.log('train/acc', acc.item())

        return {'output': tuple(o.detach() for o in output)}


    def loss_fn_exceptions(self, output, target, is_exception=None, hidden_activation=None,
                           converge_time=None):
        output, target = output[0], target[0] #unpack length-1 state tuple

        with torch.no_grad():
            if self.exception_loss_mode == 'manual':
                weights = torch.ones(output.shape[0]) #[B]
                weights[is_exception] = self.exception_loss_scaling
            elif self.exception_loss_mode == 'entropy':
                exception_signal = nc.entropy(hidden_activation)
            elif self.exception_loss_mode == 'max':
                exception_signal = -hidden_activation.max(dim=1)[0] #negative bc large max -> small loss
            elif self.exception_loss_mode == 'norm':
                exception_signal = -hidden_activation.norm(dim=1) #same
            elif self.exception_loss_mode == 'time':
                exception_signal = converge_time

            if self.exception_loss_mode != 'manual':
                #Turn `exception_signal` into (unnormalized) `weights`. We've already set `weights` if
                #we're in manual mode so only do this for the other modes
                if exception_signal.max() > self.max_exception_signal:
                    self.max_exception_signal = exception_signal.max()
                if exception_signal.min() < self.min_exception_signal:
                    self.min_exception_signal = exception_signal.min()

                #rescale to [0,1], relative to the min/max exception_signal that's ever been seen
                if self.max_exception_signal == self.min_exception_signal:
                    exception_signal = torch.ones_like(exception_signal)
                else:
                    exception_signal -= self.min_exception_signal
                    exception_signal /= (self.max_exception_signal-self.min_exception_signal)

                #rescale to [1, exc_loss_scaling]
                weights = exception_signal*(self.exception_loss_scaling-1) + 1

            weights = weights/sum(weights) #normalize to sum to 1

        loss_per_item = F.mse_loss(output, target, reduction='none').mean(1)
        loss_exc = loss_reg = None
        if is_exception is not None and is_exception.any():
            #TODO: just reweight loss_exc and loss_reg instead of each item individually if manual mode
            loss_exc = loss_per_item[is_exception].sum()
            loss_reg = loss_per_item[~is_exception].sum()
        loss = (weights*loss_per_item).sum()

        return loss, loss_reg, loss_exc


    def get_hidden_trajectory(self, batch, select_neurons=None):
        input, target, perturb_mask = batch[0:3]
        with torch.no_grad():
            state_trajectory = self(input, clamp_mask=~perturb_mask, return_mode='trajectory') #[T x ([B,N],[B,M])]

        hidden_trajectory = torch.stack([state[1] for state in state_trajectory]) #[T,B,M]
        if select_neurons is not None:
            hidden_trajectory = hidden_trajectory[:,:,select_neurons]
        return hidden_trajectory


    def plot_hidden_trajectory(self, batch, select_neurons=None):
        hidden_trajectory = self.get_hidden_trajectory(batch, select_neurons) #[T,B,M]
        n_hidden_units = hidden_trajectory.shape[-1]
        if select_neurons is None:
            select_neurons = range(n_hidden_units)
        fig, ax = plt.subplots(*plots.length_to_rows_cols(n_hidden_units), sharex=True, sharey=True)
        for i in range(n_hidden_units):
            ax.flat[i].plot(hidden_trajectory[:,:,i])
            # ax[i].set_title(f'neuron {select_neurons[i]}')
            ax.flat[i].axis('off')
        ax[-1,0].axis('on')
        ax[-1,0].spines['top'].set_visible(False)
        ax[-1,0].spines['right'].set_visible(False)
        ax[-1,0].set_xlabel('time')
        ax[-1,0].set_ylabel('softmax($h_\mu$)')
        fig.set_size_inches(7.12, 7.12)
        fig.tight_layout()
        return fig, ax



class ConvThreeLayer(AssociativeMemory):
    """ From Krotov 2021
    x: [B,Cx,L,L]
    y: [B,Cy,M,M]
    z: [B,N]
    """
    def __init__(self, x_size, x_channels, y_channels, kernel_size, z_size, stride=1,
                 dt=0.1, *args, **kwargs):
        super().__init__(dt=dt, *args, **kwargs)
        y_size = math.floor((x_size-kernel_size)/stride)+1
        self.x = nc.Layer(x_channels, x_size, x_size, dt=self.dt, tau=1)
        self.y = nc.Layer(y_channels, y_size, y_size, nonlin=nc.Softmax(beta=0.1), dt=self.dt, tau=0.2)
        self.z = nc.Layer(z_size, nonlin=nc.Softmax(), dt=self.dt, tau=0)

        self.conv = nc.Conv2d(x_channels, y_channels, kernel_size, stride)
        self.fcr = nc.Reshape(nc.Linear(math.prod(self.y.shape), z_size),
                             (math.prod(self.y.shape),))


    def default_init(self, x_init, y_init=None):
        if y_init is None:
            y_init = self.y.default_init(x_init.shape[0]).to(x_init.device)
        return x_init, y_init


    def configure_callbacks(self):
        return super().configure_callbacks() + [cb.WeightsLogger(self.sparse_log_factor)]


    def step(self, x_prev, y_prev, z_prev=None):
        #z_prev is ignored because tau_z=0 (included for function signature consistency)
        z = self.z.step(0, self.fcr(y_prev))
        y = self.y.step(y_prev, self.fcr.T(z)+self.conv(x_prev)) #use z not z_prev since tau_z=0
        x = self.x.step(x_prev, self.conv.T(y_prev))
        return x,y,z


    @staticmethod
    def infer_input_size(train_data):
        """
        train_data: (X,Y) where X is [D,Cx,L,L]
        """
        return {'x_size' : train_data[0][0].shape[-1],
                'x_channels' : train_data[0][0].shape[0]}


    def plot_weights(self, weights='weights', title='', drop_last=0, pad_nan=True):
        if weights == 'weights':
            weights = self.conv.weight.detach()
            title = f'{self.__class__.__name__} conv_weight'
        elif weights == 'grads':
            weights = self.conv.weight.grad
            title = f'{self.__class__.__name__} grad_conv_weight'
        elif not isinstance(weights, torch.Tensor):
            raise ValueError()

        rows,_ = plots.length_to_rows_cols(len(weights))
        grid = torchvision.utils.make_grid(weights.cpu(), rows, padding=1, normalize=True,
                                           pad_value=torch.tensor(float('nan')))
        grid = grid.transpose(0,1).transpose(1,2) #[C,W,H]->[W,C,H]->[W,H,C]
        fig, ax = plots.plot_matrix(grid, title=title)
        return fig, ax




###############
# Simple test #
###############
if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader

    def make_net(net_type, train_mode):
        if net_type == 'conv':
            B = 4
            x_size = 10
            x_channels = 2
            train_data = TensorDataset(torch.rand(B, x_channels, x_size, x_size),
                                       torch.rand(B, x_channels, x_size, x_size),
                                       torch.rand(B, x_channels, x_size, x_size).round().bool())
            net = ConvThreeLayer(x_size, x_channels, y_channels=3, kernel_size=3, z_size=10,
                                 train_kwargs={'mode':train_mode}, converged_thres=1e-7)
        elif net_type == 'large':
            B = 64
            N = 100
            M = 500
            train_data = TensorDataset(torch.rand(B,N),
                                       torch.rand(B,N),
                                       torch.rand(B,N).round().bool())
            net = LargeAssociativeMemory(N, M, train_kwargs={'mode':train_mode},
                                         converged_thres=1e-7)
        return net, train_data


    def train(net, train_data):
        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
        timer = pl.callbacks.Timer()
        printer = cb.Printer()
        print(f'\nStarting {net.__class__.__name__} ({net.train_mode})')
        trainer = pl.Trainer(max_steps=1,
                             logger=False,
                             enable_model_summary=False,
                             enable_progress_bar=False,
                             enable_checkpointing=False,
                             callbacks=[timer, printer],
                             # profiler='advanced',
                              )
        trainer.fit(net, train_loader)
        print(f'Time elapsed ({net.train_mode}): {timer.time_elapsed("train")}\n')


    def compare_grads(net1, net2):
        assert type(net1)==type(net2)
        for (name1, param1), (name2, param2) in zip(net1.named_parameters(), net2.named_parameters()):
            assert name1 == name2
            if param1.requires_grad and param2.requires_grad:
                grad1 = param1.grad
                grad2 = param2.grad
                assert grad1 is not None and grad2 is not None
                err = (grad1-grad2).abs().max().item()
                allclose = torch.allclose(grad1, grad2)
                print(f'{name1}.grad: max_abs_err={err}, allclose={allclose}')


    seed = torch.randint(1, 4294967295, (1,)).item() #upper bound is numpy's max seed
    for net_type in ['conv', 'large']:
        nets = []
        for train_mode in ['rbp', 'bptt']:
            pl.seed_everything(seed)
            net, train_data = make_net(net_type, train_mode)
            train(net, train_data)
            nets.append(net)
        compare_grads(*nets)
