import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##################
# Nonlinearities #
##################
class Softmax(nn.Softmax):
    def __init__(self, beta=1, train=False, dim=-1):
        assert beta>0
        super().__init__(dim=dim)

        self.softplus = nn.Softplus(beta=10)

        _beta = softplus_inv(torch.tensor(float(beta)), self.softplus.beta)
        self._beta = nn.Parameter(_beta, requires_grad=train)

        #TODO there's a built-in way: https://pytorch.org/tutorials/intermediate/parametrizations.html
        self.register_buffer('beta', torch.empty_like(self._beta))
        with torch.no_grad():
            self.beta = self.softplus(self._beta)

    def __str__(self):
        return f'Softmax(beta={self.beta:g}{", train=True" if self._beta.requires_grad else ""})'

    def forward(self, input):
        self.beta = self.softplus(self._beta)
        return super().forward(self.beta*input)

    def L(self, input):
        """Lagrangian. f(x)_i = dL/dx_i"""
        return torch.logsumexp(self.beta*input, self.dim)/self.beta #default: [B,*,N]->[B,*]


class Spherical(nn.Module):
    def __init__(self, centering=False, dim=-1):
        """If centering is True, equivalent to LayerNorm (see Tang and Kopp 2021)"""
        super().__init__()
        self.dim = dim

        if centering:
            raise NotImplementedError('Not tested')
        self.centering = centering

    def __str__(self):
        if self.centering:
            return f'Spherical(centering={self.centering})'
        return 'Spherical()'

    def forward(self, input):
        """f_i = (x_i - m)/||x-m||, where m = mean(x) if centering==True else 0"""
        if (input==0).all():
            return input
        if self.centering:
            input = input - input.mean(dim=self.dim, keepdim=True) #default: [B,*,N]-[B,*,1]
        return input / torch.linalg.vector_norm(input, dim=self.dim, keepdim=True) #default: [B,*,N]/[B,*,1]

    def L(self, input):
        """f(x)_i = dL/dx_i"""
        return torch.linalg.vector_norm(input, dim=self.dim) #default:[B,*,N]->[B,*]


class Identity(nn.Module):
    def forward(self, input):
        return input

    # def L(self, x):
    #     """L = 1/2 x^T*x --> dL/dx_i = x_i"""
    #     xx = x.transpose(-2,-1) @ x #[B,M,1]->[B,1,1] or #[M,1]->[1,1]
    #     return 0.5*xx.squeeze(-2) #[B,1,1]->[B,1] or #[1,1]->[1]


class Polynomial(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = nn.Parameter(torch.tensor(int(n)), requires_grad=False)

    def forward(self, input):
        return torch.pow(input, self.n)


class RectifiedPoly(nn.Module):
    def __init__(self, n, train=False):
        super().__init__()
        self.n = nn.Parameter(torch.tensor(float(n)), requires_grad=train)

    def forward(self, input):
        return torch.pow(torch.relu(input), self.n)

    def __str__(self):
        return f'RectifiedPoly(n={self.n}{", train=True" if self.n.requires_grad else ""})'


#########
# Layer #
#########
class Layer(nn.Module):
    def __init__(self, *shape, nonlin=Identity(), tau=1, dt=1):
        super().__init__()

        self.shape = shape
        self.nonlin = nonlin

        assert dt<=tau or tau==0
        self.tau = float(tau)
        self.dt = float(dt)
        self.eta = dt/tau if tau!=0 else 1.

    def default_init(self, batch_size, mode='zeros'):
        if mode == 'zeros':
            return torch.zeros(batch_size, *self.shape)
        else:
            raise ValueError(f'Invalid initialization mode: {mode}')

    def step(self, prev_state, total_input):
        """
        Applies nonlinearity f(.) and takes a single (discretized) step of the dynamics
            tau*df/dt = -f(t) + f( W*x(t) ) i.e.
            f(t+dt) = (1-dt/tau)*f(t) + dt/tau f( W*g(t) )

        state f(t+dt) and prev_state f(t) are firing rates (post-nonlin) of this layer
        total_input W*g(t) is the input currents (sum of scaled firing rates of all afferent neurons)
        """
        steady_state = self.nonlin(total_input)
        state = (1-self.eta)*prev_state + self.eta*steady_state
        return state

    def energy(self, input):
        """
        input x is shape [B,*]

        E = x^T*f(x) - L(x), where f_i(x) = dL/dx_i
        """
        raise NotImplementedError('Not tested')
        # activation = self.nonlin(input) #[B, *self.shape]
        # lagrangian = self.nonlin.L(input) #[B]
        # dims = tuple(range(1,len(self.shape)+1))
        # E = (activation*input).sum(dim=dims) - lagrangian
        # return E


#################################
# Transposeable transformations #
#################################
class Linear(nn.Module):
    #Want to do this but does not work to share weights...
    #class Linear(nn.Linear):
    #    def __init__(self, input_size, output_size):
    #        super().__init__(input_size, output_size, bias=False)
    #        self.T = nn.Linear(self.out_features, self.in_features, bias=False)
    #        self.T.weight.data = self.weight.data.t() #how to share weights correctly??
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) #same as nn.Linear

    def forward(self, input):
        return F.linear(input, self.weight)

    def T(self, input):
        return F.linear(input, self.weight.t())

    def energy(self, pre, post):
        """
        pre g=g(v) is presynaptic firing rate, shape [T,B,N]
        post f=f(h) is postsynaptic firing rate, shape [T,B,M]
        self.weight is shape [M,N]

        E = -g^T W f
        """
        raise NotImplementedError('Not tested')
        # [T,B,1,M]@[M,N]@[T,B,N,1]->[T,B,1,1]->[T,B]. Note this is backwards because self.weight
        # is stored as [input_dim, output_dim] following Pytorch convention
        E = -(pre.unsqueeze(-2) @ self.weight @ pre.unsqueeze(-1)).squeeze()
        return E #[T,B]


class LinearNormalized(nn.Module):
    def __init__(self, input_size, output_size, normalize_weights=False):
        #TODO: subclass Linear(), or remove it enturely and use the normalize=False default
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self._weight = nn.Parameter(torch.empty(output_size, input_size))
        nn.init.kaiming_uniform_(self._weight, a=math.sqrt(5)) #same as nn.Linear

        self.normalize_mode = normalize_weights
        if self.normalize_mode == 'rows_scaled':
            self.row_scaling = nn.Parameter(torch.ones(output_size, 1))

        #TODO there's a built-in way: https://pytorch.org/tutorials/intermediate/parametrizations.html
        self.register_buffer('weight', torch.empty_like(self._weight))
        with torch.no_grad():
            self.normalize_weights()

    def normalize_weights(self):
        if self.normalize_mode == 'frobenius':
            self.weight = self._weight / self._weight.norm()
        elif self.normalize_mode == 'rows':
            self.weight = self._weight / self._weight.norm(dim=1, keepdim=True)
        elif self.normalize_mode == 'rows_scaled':
            self.row_scaling.data = torch.clamp(self.row_scaling, min=0, max=1)
            self.weight = self._weight / self._weight.norm(dim=1, keepdim=True) * self.row_scaling
        elif self.normalize_mode is False:
            self.weight = self._weight
        else:
            raise ValueError('Invalid weight normalization mode')

    def forward(self, input):
        self.normalize_weights()
        return F.linear(input, self.weight)

    def T(self, input):
        self.normalize_weights()
        return F.linear(input, self.weight.t())

    def energy(self, input):
        #TODO: remove after subclassing Linear()
        raise NotImplementedError()


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.T = nn.ConvTranspose2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, bias=False)
        self.T.weight = self.weight


class AvgPool2d(nn.AvgPool2d):
    pass #TODO


class Reshape(nn.Module):
    def __init__(self, function, new_shape):
        """
        function: a transposeable transformation
        new_shape: must be the shape that <function> accepts, excluding the batch dimension"""
        super().__init__()
        self.function = function
        self.new_shape = new_shape
        self.old_shape = None #dynamically computed during forward

    def forward(self, input):
        """Reshapes feedforward input to new_shape before putting it through the function"""
        self.old_shape = input.shape
        batch_size = input.shape[0]
        input = input.reshape(batch_size, *self.new_shape)
        return self.function(input)

    def T(self, input):
        """Puts the feedback input through the function and reshapes it to what it
        was before the feedforward reshaping"""
        output = self.function.T(input)
        return output.reshape(*self.old_shape)

#########
# Utils #
#########
class HotSwapParameter():
    """Context manager to temporarily set a parameter to a desired value, e.g. runs `net(input)` with
     `net.param` set to `value`, then puts `net.param` back to whatever it was before:

    with HotSwapParameter(net.param, value):
        output = net(input)
    """
    def __init__(self, param, temp_value):
        self.param = param
        self.temp_value = temp_value
        self.original_value = None

    def __enter__(self):
        self.original_value = self.param.data.clone()
        self.param.data = self.temp_value

    def __exit__(self, *args):
        self.param.data = self.original_value


def softplus_inv(x, beta=1, threshold=20):
    if x*beta > threshold:
        return x
    return (torch.log(torch.expm1(beta*x)))/beta


def entropy(x):
    """x.shape = [B,N]. Every row must sum to 1."""
    assert (x>0).all()
    assert torch.allclose(x.sum(dim=1), torch.ones(x.shape[0])) #allclose to allow for numerical error
    return -(x*torch.log2(x)).sum(dim=1)
