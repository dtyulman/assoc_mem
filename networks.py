import math, types
from copy import deepcopy

import torch
import torchvision
from torch.nn import functional as F
import pytorch_lightning as pl

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


    def forward(self, *init_state, clamp_mask=None, debug=False):
        """
        init_state: list of initial states for each layer in network. Set list entry
            to None to use the layer's default. At least one entry must be provided.
        clamp_mask: same shape as init_state[0], selects which entries of the state to clamp
        """
        state = self.default_init(*init_state)
        if 'clamp' in self.input_mode:
            #TODO: this assumes state[0] is the (perturbed) "input". In general might be
            # any/some/all of the entries in the state tuple
            clamp_values = state[0][clamp_mask]

        if debug: state_history = [state]
        for t in range(1,self.max_steps+1):
            prev_state = state
            state = self.step(*prev_state)
            if 'clamp' in self.input_mode: #TODO: same as above
                state[0][clamp_mask] = clamp_values
            if debug: state_history.append(state)
            residual = self.get_residual(state, prev_state)
            if residual <= self.converged_thres:
                break
        if residual > self.converged_thres and self.warn_not_converged:
            print(f'FP not converged: t={t}, residual={residual:.1e}, thres={self.converged_thres}')

        self.log('steps_to_converge', t)

        if debug:
            return state_history
        return state


    @staticmethod
    def flatten(state):
        return torch.cat([s.flatten() for s in state])


    @staticmethod
    def get_residual(state, prev_state):
        with torch.no_grad():
            state = AssociativeMemory.flatten(state)
            prev_state = AssociativeMemory.flatten(prev_state)
            return ((state - prev_state).norm()/state.norm()).item()


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
            def training_forward(self, input, clamp_mask=None):
                return self(input, clamp_mask=clamp_mask)

        elif mode == 'rbp':
            self.rbp_max_steps = kwargs.pop('max_steps', 5000)
            self.rbp_converged_thres = kwargs.pop('converged_thres', 1e-6)
            def pack(*state):
                if len(state)==1:
                    return state[0], None
                batch_size = state[0].shape[0]
                unpacked_shape = [s.shape[1:] for s in state]
                packed_state = torch.cat([s.reshape(batch_size,-1) for s in state], dim=1)
                return packed_state, unpacked_shape

            def unpack(packed_state, unpacked_shape):
                if unpacked_shape is None:
                    return (packed_state,)
                batch_size = packed_state.shape[0]
                i = 0
                unpacked = []
                for shape in unpacked_shape:
                    j = i+math.prod(shape)
                    unpacked.append(packed_state[:,i:j].reshape(batch_size, *shape))
                    i = j
                return unpacked

            def training_forward(self, input, clamp_mask=None):
                with torch.no_grad():
                    fp = self(input, clamp_mask=clamp_mask)
                fp,unpacked_shape = pack(*self.step(*fp)) #packing so that autograd gets engaged properly

                fp_in = fp.detach().clone().requires_grad_()
                fp_out,_ = pack(*self.step(*unpack(fp_in,unpacked_shape)))
                def backward_hook(grad):
                    new_grad = grad
                    for t in range(1,self.rbp_max_steps+1):
                        new_grad_prev = new_grad
                        new_grad = torch.autograd.grad(fp_out, fp_in, new_grad_prev, retain_graph=True)[0] + grad
                        rel_res = ((new_grad - new_grad_prev).norm()/new_grad.norm()).item()
                        if rel_res <= self.rbp_converged_thres:
                            break
                    if t > self.rbp_max_steps:
                        print(f'BP not converged: t={t}, res={rel_res}')
                    return new_grad
                fp.register_hook(backward_hook)
                return unpack(fp,unpacked_shape)

        elif self.train_mode == 'rbp-1h':
            def training_forward(input, clamp_mask=None):
                pass
        else:
            raise ValueError('Invalid train_mode: {self.train_mode}')

        self.training_forward = types.MethodType(training_forward, self)



class LargeAssociativeMemory(AssociativeMemory):
    """From Krotov and Hopfield 2020
    x: [B,N] (feature)
    y: [B,M] (hidden)
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
        self.save_hyperparameters()

        self.feature = nc.Layer(input_size, nonlin=input_nonlin, tau=float(tau), dt=self.dt)
        self.fc = nc.LinearNormalized(input_size, hidden_size, normalize_weights=normalize_weights)
        self.hidden = nc.Layer(hidden_size, nonlin=hidden_nonlin, tau=0, dt=self.dt)

        self.rescale_grads = rescale_grads


    def on_after_backward(self):
        """Optionally rescales gradient of each hidden unit's weights s.t. the max grad (across
        all afferent weights for that neuron) is equal to 1"""
        #TODO: can we move this to self.fc's class to make more general?
        if self.rescale_grads:
            #weight.shape = [hidden_size, input_size]
            self.fc._weight.grad /= self.fc._weight.grad.abs().max(dim=1, keepdim=True)[0]


    def step(self, x_prev):
        y = self.hidden.step(0, self.fc(x_prev))
        x = self.feature.step(x_prev, self.fc.T(y))
        return (x,) #wrap in tuple b/c base class requires returning tuple of states


    @staticmethod
    def infer_input_size(train_data):
        """train_data: (X,Y) where X is [D,N]"""
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
                 exception_loss_scaling=1,
                  *args, **kwargs):
        """
        exception_loss_scaling: Loss weighting for exceptions is `exception_loss_scaling` times
            [loss for non-exceptions]. ALl loss weightings are normalized such that they sum to 1.
            Remember to run net.set_exception_loss_scaling() after initializing the network!
        """
        self.exceptions = set(exceptions)
        self.exception_loss_scaling = exception_loss_scaling
        #remember to run net.set_exception_loss_scaling() after initializing the network!


        self.beta_exception = torch.tensor(float(beta_exception)) if beta_exception is not None else None
        assert not train_beta or beta_exception is None, "Can't use dynamic beta if training beta0"
        super().__init__(*args, hidden_nonlin=nc.Softmax(beta=beta, train=train_beta), **kwargs)


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



    def training_step(self, batch, batch_idx):
        input, target, perturb_mask, label = batch
        if isinstance(target, torch.Tensor):
            target = (target,) #output is tuple, make sure target is too
        output = self.training_forward(input, clamp_mask=~perturb_mask)

        #optionally re-run network with different beta for exceptions
        is_exception = torch.tensor([l.item() in self.exceptions for l in label])
        if self.beta_exception is not None and is_exception.any():
            with nc.HotSwapParameter(self.hidden.nonlin.beta, self.beta_exception):
                #remember to unpack length-1 output state tuple
                output[0][is_exception] = self.training_forward(
                    input[is_exception], clamp_mask=~perturb_mask[is_exception])[0]

        loss = self.loss_fn(output, target, is_exception)
        acc = self.acc_fn(output, target)

        self.log('train/loss', loss.item())
        self.log('train/acc', acc.item())

        return {'loss': loss,
                'output': tuple(o.detach() for o in output)}


    def loss_fn(self, output, target, is_exception):
        output, target = output[0], target[0] #unpack length-1 state tuple

        weights = torch.ones(output.shape[0], 1)
        weights[is_exception] = self.exception_loss_scaling
        weights = weights/sum(weights)

        loss = (weights*F.mse_loss(output, target, reduction='none')).sum()
        return loss


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


    def step(self, x_prev, y_prev):
        z = self.z.step(0, self.fcr(y_prev))
        y = self.y.step(y_prev, self.fcr.T(z)+self.conv(x_prev)) #use z not z_prev since tau_z=0
        x = self.x.step(x_prev, self.conv.T(y_prev))
        return x, y

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


class Hierarchical(AssociativeMemory):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(layers) > 1
        self.layers = layers
        self.num_layers = len(layers)

    def default_init(self, *init_state):
        raise NotImplementedError() #TODO

    def step(self, *prev_state):
        for i in range(self.num_layers):
            raise NotImplementedError() #TODO



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
