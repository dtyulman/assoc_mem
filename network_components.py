import math
import torch
import torch.nn as nn
import torch.nn.functional as F

##################
# Nonlinearities #
##################
class Softmax(nn.Softmax):
    def __init__(self, beta=1, train=False, dim=1):
        super().__init__(dim=dim)
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=train)

    def __str__(self):
        return f'Softmax(beta={self.beta}{", train=True" if self.beta.requires_grad else ""})'

    def forward(self, input):
        return super().forward(self.beta*input)


class Spherical(nn.Module):
    def forward(self, input):
        return input / torch.linalg.vector_norm(input, dim=-2, keepdim=True)


class Identity(nn.Module):
    def forward(self, input):
        return input


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
        steady_state = self.nonlin(total_input)
        state = (1-self.eta)*prev_state + self.eta*steady_state
        return state


#################################
# Transposeable transformations #
#################################
class Linear(nn.Module):
#Want to do this but does not work to share weights...
#class Linear(nn.Linear):
#    def __init__(self, input_size, output_size):
#        super().__init__(input_size, output_size, bias=False)
#        self.T = nn.Linear(self.out_features, self.in_features, bias=False)
#        self.T.weight.data = self.weight.data.t() #how to share weights correctly?
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
