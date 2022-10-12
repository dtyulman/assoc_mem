from pytorch_lightning import Callback

import plots

#############
# Callbacks #
#############
class Printer(Callback):
    def __init__(self, print_every=None):
        """
        print_every: int or None. If None, defaults to trainer.log_every_n_steps
        """
        self.print_every = print_every

    def setup(self, trainer, module, stage=None):
        if self.print_every is None:
            self.print_every = trainer.log_every_n_steps

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
        if trainer.global_step % self.print_every == 0:
            string = f'it={trainer.global_step}, ep={trainer.current_epoch}, bx={batch_idx}'
            if 'train/loss' in trainer.callback_metrics:
                loss = trainer.callback_metrics['train/loss'].item()
                string += f', loss={loss:.4e}'
            if 'train/acc' in trainer.callback_metrics:
                acc = trainer.callback_metrics['train/acc'].item()
                string += f', acc={acc}'
            print(string)


class ParamsLogger(Callback):
    def on_after_backward(self, trainer, module):
        for name, param in module.named_parameters():
            if param.numel() > 1:
                module.log(f'params/{name} (norm)', param.norm().item())
                module.log(f'params/{name} (mean)', param.mean().item())
                module.log(f'params/{name} (var)', param.var().item())
            else:
                module.log(f'params/{name}', param.item())

            if param.requires_grad:
                if param.numel() > 1:
                    module.log(f'grads/{name} (norm)', param.grad.norm().item())
                    module.log(f'grads/{name} (mean)', param.grad.mean().item())
                    module.log(f'grads/{name} (var)', param.grad.var().item())
                else:
                    module.log(f'grads/{name}', param.grad.item())


class SparseLogger(Callback):
    def __init__(self, sparse_log_factor=5):
        self.sparse_log_factor = sparse_log_factor

    def setup(self, trainer, module, stage=None):
        self.sparse_log_every = trainer.log_every_n_steps*self.sparse_log_factor


class WeightsLogger(SparseLogger):
    """requires module.plot_weights(weights:['weights'|'grad'])->(fig,*)
    """
    def on_after_backward(self, trainer, module):
        if (trainer.global_step+1) % self.sparse_log_every == 0:
            with plots.NonInteractiveContext():
                fig = module.plot_weights()[0]
                trainer.logger.experiment.add_figure('params/weight', fig,
                                                     global_step=trainer.global_step)

                fig = module.plot_weights(weights='grads')[0]
                trainer.logger.experiment.add_figure('grad/weight', fig,
                                                     global_step=trainer.global_step)


class OutputsLogger(SparseLogger):
    """requires train_data.plot_batch(input,target,output)->(fig,*)
    and module.training_step(...)->{'output':module(input),*}
    """
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (trainer.global_step+1) % self.sparse_log_every == 0:
            input, target = batch[0], batch[1]
            with plots.NonInteractiveContext():
                #TODO: this hard-codes the zeroth item of the final fixed point
                #state tuple as the output. In general might be any/some/all of
                #the entries in the state tuple
                fig, ax = trainer.train_dataloader.dataset.datasets \
                            .plot_batch(inputs=input, targets=target, outputs=outputs['output'][0])
            trainer.logger.experiment.add_figure('sample_batch', fig,
                                                 global_step=trainer.global_step)
