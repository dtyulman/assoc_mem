from copy import deepcopy
import math, time

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

##################
# Nonlinearities #
##################
class Softmax(nn.Softmax):
    def __init__(self, *args, beta=1, train_beta=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=train_beta)

    def forward(self, input):
        return super().forward(self.beta*input)


class Spherical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return  input / torch.linalg.vector_norm(input, dim=-2, keepdim=True)

############
# Networks #
############
class TwoLayerMem(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 input_nonlin = lambda x: x,
                 hidden_nonlin = Softmax(beta=1, train_beta=False, dim=1),
                 tau=1.,
                 dt=0.1,
                 converged_thres=1e-10,
                 max_steps=1e5):
        super().__init__()

        self.N = input_size
        self.M = hidden_size
        self.g = input_nonlin
        self.f = hidden_nonlin

        self.W = nn.Parameter(torch.empty((self.M, self.N)))
        nn.init.kaiming_normal_(self.W)

        self.dt = float(dt)
        self.tau = float(tau)
        self.eta = self.dt/self.tau
        self.converged_thres = converged_thres
        self.max_steps = int(max_steps)

    def forward(self, v0):
        state = self.g(v0)
        for t in range(self.max_steps):
            prev_state = state
            state = self.step(prev_state)
            with torch.no_grad():
                step_size = (state - prev_state).norm()
                if step_size <= self.converged_thres:
                    print(f'FF converged: t={t} step={step_size.item()}')
                    break
        if t >= self.max_steps-1:
            print(f'FF not converged: t={t}, step={step_size}, thres={self.converged_thres}')
        return state

    def step(self, g_prev):
        f = self.f(self.W @ g_prev)
        g = self.g(self.W.t() @ f)

        g = (1-self.eta)*g_prev + self.eta*g
        return g


class ConvMem(nn.Module):
    def __init__(self,
                 x_size,
                 x_channels,
                 y_channels,
                 z_size,
                 kernel_size,
                 stride = 1,
                 y_nonlin = Softmax(beta=1, train_beta=False, dim=1),
                 z_nonlin = Softmax(beta=1, train_beta=False, dim=1),
                 tau_x = 1,
                 tau_y = 0.1,
                 dt = 0.1,
                 converged_thres = 1e-10,
                 max_steps = 5000):
        super().__init__()
        self.L = x_size
        self.M = math.floor((x_size-kernel_size)/stride)+1
        self.N = z_size
        self.Cx = x_channels
        self.Cy = y_channels
        self.K = kernel_size
        self.S = stride

        self.f = y_nonlin
        self.g = z_nonlin

        self.conv = nn.Conv2d(self.Cx, self.Cy, self.K, self.S, bias=False)
        self.convT = nn.ConvTranspose2d(self.Cy, self.Cx, self.K, self.S, bias=False)
        self.convT.weight = self.conv.weight

        self.W = nn.Parameter(torch.empty(self.N, self.Cy*self.M*self.M)) #[N, CyMM]
        nn.init.kaiming_normal_(self.W)

        self.dt = float(dt)
        self.tau_x = float(tau_x)
        self.tau_y = float(tau_y)
        self.eta_x = self.dt/self.tau_x
        self.eta_y = self.dt/self.tau_y
        self.converged_thres = converged_thres
        self.max_steps = int(max_steps)


    def fc(self, f):
        #f is [B,Cy,M,M]
        z = self.W @ f.reshape(f.shape[0], self.Cy*self.M*self.M, 1) #[N,CyMM]@[B,CyMM,1]->[B,N,1]
        return z

    def fcT(self, g):
        y = self.W.t() @ g #[CyMM,N]@[B,N,1]->[B,CyMM,1]
        return y.reshape(g.shape[0], self.Cy, self.M, self.M) #[B,Cy,M,M]

    def _step(self, x_prev, f_prev):
        g = self.g( self.fc(f_prev) )
        f = self.f( self.fcT(g) + self.conv(x_prev) )

        #THIS IS THE BUG
        x = self.convT(f_prev) #this is technically the correct dynamics
        # x = self.convT(f) #this matches bptt/rbp with tuples

        f = (1-self.eta_y)*f_prev + self.eta_y*f
        x = (1-self.eta_x)*x_prev + self.eta_x*x
        return x,f

    def pack(self, x, f):
        B = x.shape[0]
        return torch.cat([x.reshape(B,-1), f.reshape(B,-1)], dim=1)

    def unpack(self, s):
        B = s.shape[0]
        len_x = self.Cx*self.L*self.L
        x = s[:,:len_x].reshape(B,self.Cx,self.L,self.L)
        f = s[:,len_x:].reshape(B,self.Cy,self.M,self.M)
        return x,f

    def init_f0(self, B):
        return torch.zeros(B,self.Cy,self.M,self.M)

    ### Using tuples
    def forward(self, x0, f0=None):
        if f0 is None:
            f0 = self.init_f0(x0.shape[0])
        state = (x0, f0)
        for t in range(self.max_steps):
            prev_state = state
            state = self.step(*prev_state)
            with torch.no_grad():
                step_size = tuple_residual(state, prev_state)
                if step_size <= self.converged_thres:
                    print(f'FF converged: t={t} step={step_size}')
                    break
        if t >= self.max_steps-1:
            print(f'FF not converged: t={t}, step={step_size}, thres={self.converged_thres}')
        return state

    def step(self, x_prev, f_prev):
        return self._step(x_prev, f_prev)


    # ### Using packing
    # def forward(self, x0):
    #     f0 = self.init_f0(x0.shape[0])
    #     state = self.pack(x0, f0) #([B,Cx,L,L],[B,Cy,M,M])
    #     for t in range(self.max_steps):
    #         prev_state = state
    #         state = self.step(prev_state)
    #         with torch.no_grad():
    #             step_size = (state - prev_state).norm()
    #             if step_size <= self.converged_thres:
    #                 print(f'FF converged: t={t} step={step_size.item()}')
    #                 break
    #     if t >= self.max_steps-1:
    #         print(f'FF not converged: t={t}, step={step_size}, thres={self.converged_thres}')
    #     return state

    # def step(self, s_prev):
    #     x_prev, f_prev = self.unpack(s_prev)
    #     x,f = self._step(x_prev, f_prev)
    #     return self.pack(x,f)




#########
# Train #
#########
class Trainer():
    def __init__(self, net, train_loader):
        self.net = net
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.loss_fn = nn.MSELoss()

    def train(self, epochs=1):
        for epoch in range(epochs):
            for i,(input,target) in enumerate(self.train_loader):
                output = self.evaluate(input)

                ### Using tuples doesnt propagate gradient
                # output = output[0]

                ### Using tuples computes grad but wrong because bw hook only
                ### updates grad_x, but grad_y is still (incorrectly) zero
                # target = self.net.pack(target, output[1])
                # output = self.net.pack(*output)

                ### Using packing
                # output = self.net.unpack(output)[0]

                ### Using packing but only inside evaluate
                output = output[0]


                loss = self.loss_fn(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, input):
        return self.net(input)


class RBPTrainer(Trainer):
    def __init__(self, *args, max_steps=5000, converged_thres=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = int(max_steps)
        self.converged_thres = converged_thres

    # def evaluate(self, input):
    #     with torch.no_grad():
    #         fp = to_tuple(self.net(input))
    #     fp = to_tuple(self.net.step(*fp))

    #     fp_in = tuple_copy_require_grad(fp)
    #     fp_out = to_tuple(self.net.step(*fp_in))
    #     def backward_hook(grad):
    #         grad = (grad,) + tuple(torch.zeros_like(fp_i) for fp_i in fp_in[1:])
    #         new_grad = grad
    #         for t in range(self.max_steps):
    #             new_grad_prev = new_grad
    #             new_grad = torch.autograd.grad(fp_out, fp_in, new_grad_prev, retain_graph=True)
    #             new_grad = tuple_sum(new_grad, grad)
    #             rel_res = tuple_residual(new_grad, new_grad_prev)
    #             if rel_res <= self.converged_thres:
    #                 print(f'BP converged: it={t} res={rel_res}')
    #                 break
    #         if t >= self.max_steps-1:
    #             print(f'BP not converged: t={t}, res={rel_res}')
    #         return new_grad[0]

    #     fp[0].register_hook(backward_hook)
    #     return fp #using tuples
    #     # return fp[0] #using packing


    ### Using packing but only inside evaluate, (tuples in network)
    def evaluate(self, input):
        with torch.no_grad():
            fp = to_tuple(self.net(input))
        fp = to_tuple(self.net.step(*fp))

        fp_in = self.net.pack(*tuple_copy_require_grad(fp))
        fp_out = self.net.pack(*to_tuple(self.net.step(*self.net.unpack(fp_in))))
        def backward_hook(grad):
            new_grad = grad
            for t in range(self.max_steps):
                new_grad_prev = new_grad
                new_grad = torch.autograd.grad(fp_out, fp_in, new_grad_prev, retain_graph=True)[0] + grad
                rel_res = ((new_grad - new_grad_prev).norm()/new_grad.norm()).item()
                if rel_res <= self.converged_thres:
                    print(f'BP converged: it={t} res={rel_res}')
                    break
            if t >= self.max_steps-1:
                print(f'BP not converged: t={t}, res={rel_res}')
            return new_grad

        fp = self.net.pack(*fp)
        fp.register_hook(backward_hook)
        return self.net.unpack(fp)

###########
# Helpers #
###########
def to_tuple(tensor_or_tensors):
    if isinstance(tensor_or_tensors, torch.Tensor):
        return (tensor_or_tensors,)
    return tuple(tensor_or_tensors)

def tuple_to_vec(seq_of_tensors):
    return torch.cat([v.flatten() for v in seq_of_tensors])

def tuple_residual(tup, tup_prev):
    vec = tuple_to_vec(tup)
    vec_prev = tuple_to_vec(tup_prev)
    return ((vec - vec_prev).norm()/vec.norm()).item()

def tuple_sum(tup1, tup2):
    return tuple(t1+t2 for t1,t2 in zip(tup1, tup2))

def tuple_copy_require_grad(tup):
    return tuple(t.detach().clone().requires_grad_() for t in tup)

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



#%%
seed = torch.random.seed()
torch.random.manual_seed(seed)

B = 1

# N = 50
# M = 55
# train_data = TensorDataset(torch.rand(D,N,1), torch.rand(D,N,1))
# net_bptt = TwoLayerMem(N, M, input_nonlin=Spherical(), converged_thres=1e-20)

L = 20
Cx = 3
Cy = 10
K = 5
S = 1
M = math.floor((L-K)/S)+1
N = 100
train_data = TensorDataset(torch.rand(B,Cx,L,L), torch.rand(B,Cx,L,L))
train_loader = DataLoader(train_data, batch_size=B, shuffle=False)

net_bptt = ConvMem(L, Cx, Cy, N, K, converged_thres=0)
net_rbp = deepcopy(net_bptt)

torch.random.manual_seed(seed)
with Timer('RBP Train'):
    trainer_rbp = RBPTrainer(net_rbp, train_loader)
    trainer_rbp.train()

torch.random.manual_seed(seed) #reseed because f0 is random
with Timer('BPTT Train'):
    trainer_bptt = Trainer(net_bptt, train_loader)
    trainer_bptt.train()


print('conv.w.grad mean err=',(net_bptt.conv.weight.grad-net_rbp.conv.weight.grad).abs().mean().item(),
      'allclose=', torch.allclose(net_bptt.conv.weight.grad, net_rbp.conv.weight.grad))
print('W.grad mean err=',(net_bptt.W.grad-net_rbp.W.grad).abs().mean().item(),
      'allclose=', torch.allclose(net_bptt.W.grad, net_rbp.W.grad))
