import math, warnings
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

    def __repr__(self):
        return f'Softmax(beta={self.beta:g}{", train=True" if self._beta.requires_grad else ""})'

    def forward(self, input):
        self.beta = self.softplus(self._beta)
        return super().forward(self.beta*input)

    def L(self, input):
        """Lagrangian. f(x)_i = dL/dx_i"""
        input = nan2minf(input)
        return torch.logsumexp(self.beta*input, self.dim)/self.beta #default: [B,*,N]->[B,*]


class Spherical(nn.Module):
    def __init__(self, centering=False, dim=-1):
        """If centering is True, equivalent to LayerNorm (see Tang and Kopp 2021)"""
        super().__init__()
        self.dim = dim

        if centering:
            raise NotImplementedError('Not tested')
        self.centering = centering

    def __repr__(self):
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
        input = nan2zero(input)
        return torch.linalg.vector_norm(input, dim=self.dim) #default:[B,*,N]->[B,*]


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def __repr__(self):
        return 'Sigmoid()'

    def forward(self, input):
        return torch.sigmoid(input)

    def L(self, input):
        input = nan2minf(input)
        return self.softplus(input).sum(dim=-1)


class Identity(nn.Module):
    def forward(self, input):
        #clone in case downstream does in-place op on f(x), which should not change x
        return input.clone()

    def L(self, input):
        """L = 1/2 x^T*x --> dL/dx_i = x_i"""
        input = nan2zero(input)
        return torch.pow(input,2).sum(dim=-1)/2 #[T,B,N]/[B,N]/[N] -> [T,B]/[B]/[1]


class Polynomial(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = nn.Parameter(torch.tensor(int(n)), requires_grad=False)

    def __repr__(self):
        return f'Polynomial(n={self.n}{", train=True" if self.n.requires_grad else ""})'

    def forward(self, input):
        return torch.pow(input, self.n)

    def L(self, input):
        """L =  1/(n+1) \sum_i x_i^n+1 --> dL/dx_i = x_i^n"""
        input = nan2zero(input)
        return torch.pow(input, self.n+1).sum(dim=-1)/(self.n+1) #[T,B,N]/[B,N]/[N] -> [T,B]/[B]/[1]


class RectifiedPoly(nn.Module):
    def __init__(self, n, train=False):
        super().__init__()
        self.n = nn.Parameter(torch.tensor(float(n)), requires_grad=train)

    def __repr__(self):
        return f'RectifiedPoly(n={self.n}{", train=True" if self.n.requires_grad else ""})'

    def forward(self, input):
        return torch.pow(torch.relu(input), self.n)


ELEMENTWISE_NONLINS = (Sigmoid, Identity, Polynomial, RectifiedPoly)


#########
# Layer #
#########
class Layer(nn.Module):
    def __init__(self, *shape, nonlin=Identity(), state_mode='rates', tau=1, dt=1):
        super().__init__()

        self.shape = shape
        self.nonlin = nonlin
        self.state_mode = state_mode

        assert dt<=tau or tau==0
        self.tau = float(tau)
        self.dt = float(dt)
        self.eta = dt/tau if tau!=0 else 1.

    def default_init(self, batch_size, mode='zeros'):
        if mode == 'zeros':
            return torch.zeros(batch_size, *self.shape)
        else:
            raise ValueError(f'Invalid initialization mode: {mode}')

    def step(self, prev_state, total_input, clamp_mask=None, clamp_values=None):
        """
        if state_mode == 'rates':
            Applies nonlinearity F(.) and takes a single (discretized) step of the dynamics
                tau*df/dt = -f(t) + F(W*g(t))
                i.e. f(t+dt) = (1-dt/tau) f(t) + dt/tau F(W*g(t))

            state f(t+dt) and prev_state f(t) are firing rates (post-nonlin) of this layer
            total_input W*g(t)+I is the input currents (sum of scaled firing rates of all afferent neurons)

        elif state_mode == 'currents':
            tau*dx/dt = -x(t) + W*G(y(t))
            i.e. x(t+dt) = (1-dt/tau) x(t) + dt/tau W*G(y(t))

            state x(t+dt) and prev_state x(t) are currents (pre-nonlin)
            total_input W*G(y(t))+I as above, but note that you'll need to apply the other layer's nonlin first
        """
        if self.state_mode == 'rates':
            steady_state = self.nonlin(total_input)
        elif self.state_mode == 'currents':
            steady_state = total_input

        state = (1-self.eta)*prev_state + self.eta*steady_state
        if clamp_mask is not None and clamp_values is not None:
            state[clamp_mask] = clamp_values
        return state


    def energy(self, input, external=0):
        """
        input x is [T,B,*self.shape] e.g [T,B,N] or [T,B,C,L,L]
         or [B,*self.shape] e.g [B,N] or [B,C,L,L]
         or [*self.shape] e.g [N] or [C,L,L]

        E = (x-I)^T * f(x) - L(x), where f_i(x) = dL/dx_i and I is external input current
        """

        assert external is 0 or external.shape==input.shape #noqa

        activation = self.nonlin(input)
        lagrangian = self.nonlin.L(input)

        n_layer_dims = len(self.shape)
        n_input_dims = len(input.shape)
        layer_dims = tuple(range(n_input_dims-n_layer_dims, n_input_dims))

        E = ((input-external)*activation).nansum(dim=layer_dims) - lagrangian
        return E


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
        pre g=g(v) is presynaptic firing rate, shape [T,B,N] or [B,N] or [N]
        post f=f(h) is postsynaptic firing rate, shape [T,B,M] or [B,M] or [M]
        self.weight is shape [M,N]

        E = -g^T W f
        """
        # [T,B,1,M]@[M,N]@[T,B,N,1]->[T,B,1,1]->[T,B]. Note this is backwards because self.weight
        # is stored as [input_dim, output_dim] following pytorch convention
        return -(post.unsqueeze(-2) @ self.weight @ pre.unsqueeze(-1)).squeeze(-1).squeeze(-1) #[T,B]


class LinearNormalized(nn.Module):
    def __init__(self, input_size, output_size, normalize_weights=False):
        #TODO: subclass Linear(), or remove it entirely and use the normalize=False default
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

    def energy(self, pre, post):
        #TODO: remove after subclassing Linear()
        """
        pre g=g(v) is presynaptic firing rate, shape [T,B,N] or [B,N] or [N]
        post f=f(h) is postsynaptic firing rate, shape [T,B,M] or [B,M] or [M]
        self.weight is shape [M,N]

        E = -g^T W f
        """
        # [T,B,1,M]@[M,N]@[T,B,N,1]->[T,B,1,1]->[T,B]. Note this is backwards because self.weight
        # is stored as [input_dim, output_dim] following pytorch convention
        return -(post.unsqueeze(-2) @ self.weight @ pre.unsqueeze(-1)).squeeze(-1).squeeze(-1) #[T,B]


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
    """x.shape = [B,N]"""
    assert (x>0).all(), 'All entries must be positive'
    assert torch.allclose(x.sum(dim=1), torch.ones(x.shape[0])), 'Every row must sum to 1' #allclose to allow for numerical error
    return -(x*torch.log2(x)).sum(dim=1)

def nan2minf(x):
    x = x.clone()
    x[x.isnan()] = -float('inf')
    return x

def nan2zero(x):
    x = x.clone()
    x[x.isnan()] = 0
    return x
