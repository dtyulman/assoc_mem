import warnings, math
from itertools import chain, combinations

import torch
import torch.nn.functional as F
from torch import nn


######################
### Nonlinearities ###
######################
class Nonlinearity(nn.Module):
    """Base class"""
    def __init__(self):
        #for every trainable parameter in the nonlinearity, must add a function
        #called D_<param>(x) that computes partial-f/partial-<param>
        super().__init__()

    def __call__(self, x, *args, **kwargs):
        return self._function(x, *args, **kwargs)

    def _function(self, x):
        raise NotImplementedError()

    def L(self, x):
        """Lagrangian function of the neural activity,
        defined s.t. dL/dx_i = f(x)_i """
        raise NotImplementedError()

    def J(self, x):
        """Jacobian of the function f: J_ij = df_i/dx_j, evaluated at x"""
        raise NotImplementedError()



class Power(Nonlinearity):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def _function(self, x):
        return torch.pow(x, self.n)

    def L(self, x):
        """Lagrangian function of the neural activity,
        defined s.t. dL/dx_i = f(x)_i """
        raise NotImplementedError()

    def J(self, x):
        """Jacobian of the function f: J_ij = df_i/dx_j, evaluated at x"""
        raise NotImplementedError()

    def D_n(self, x):
        """ del f / del n = #TODO
        Partial derivative of f wrt parameter n"""
        raise NotImplementedError()


class Hardmax(Nonlinearity):
    """Softmax with beta = inf
    i.e. zero-order approximation of Softmax
    """
    def _function(self, x):
        argmax = torch.argmax(x, dim=-2)
        onehot = F.one_hot(argmax, num_classes=x.shape[-2]).transpose(-2,-1).to(torch.get_default_dtype())
        return onehot


    def J(self, x):
        """Jacobian of the function f: J_ij = df_i/dx_j, evaluated at x"""
        if len(x.shape) == 3:
            return torch.zeros(x.shape[0], x.shape[-2], x.shape[-2])
        elif len(x.shape) == 2:
            return torch.zeros(x.shape[-2], x.shape[-2])
        else:
            raise ValueError()


class Softmax_1(Hardmax):
    """
    First-order approximation of the Softmax function
    """
    def __init__(self, beta=1., train=False):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=train)

    def __str__(self):
        return f'Softmax_1(beta={self.beta}, train={self.beta.requires_grad})'

    def __repr__(self):
        return self.__str__()

    def _function(self, x, f0=None, eps=None):
        if f0 is None:
            f0 = self._zeroth_order(x) #zeroth order correction
        if eps is None:
            eps = self._eps(x) #perturbative variable
        return (1-eps.sum(dim=-2, keepdim=True))*f0 + eps #f1 = f0 + (I-E)*eps

    def _eps(self, x):
        return torch.exp(self.beta*(x - x.max(dim=-2, keepdim=True)[0]))

    def _zeroth_order(self, x):
        return super()._function(x)

    def _FF(self, f0, eps):
        FF = eps.squeeze(-1).diag_embed()
        FF -= eps @ f0.transpose(-2,-1)
        FF -= f0 @ eps.transpose(-2,-1)
        FF += eps.sum(dim=-2, keepdim=True) * f0 @ f0.transpose(-2,-1)
        return FF

    def J(self, x, f0=None, eps=None):
        if f0 is None:
            f0 = self._zeroth_order(x)
        if eps is None:
            eps = self._eps(x)
        return self.beta*self._FF(f0, eps)

    def D_beta(self, x, f0=None, eps=None, FF=None):
        if f0 is None:
            f0 = self._zeroth_order(x)
        if eps is None:
            eps = self._eps(x)
        if FF is None:
            FF = self._FF(f0, eps)
        return FF @ x


class Softmax(Nonlinearity):
    def __init__(self, beta=1., train=False):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=train)

    def __str__(self):
        return f'Softmax(beta={self.beta}, train={self.beta.requires_grad})'

    def __repr__(self):
        return self.__str__()

    def _function(self, x):
        """f_i = softmax(x)_i"""
        return F.softmax(self.beta*x, dim=-2) #[B,N,1] or [N,1]

    def L(self, x):
        """f(x)_i = dL/dx_i"""
        return torch.logsumexp(self.beta*x, -2)/self.beta #[B,N,1]->[B,1] or #[N,1]->[1]

    def J(self, x, f=None, FF=None):
        """J_f =  beta*(diag(f(x)) - f(x)*f(x)^T) Jacobian of f evaluted at x.
        Pass in FF=self._FF(x) if you have it cached to avoid re-computing"""
        if FF is None:
            FF = self._FF(x, f)
        return self.beta * FF #[B,N,N]

    def D_beta(self, x, f=None, FF=None):
        """ del f / del beta = (diag(f(x)) - f(x)*f(x)^T) * x
        Partial derivative of f wrt parameter beta. Pass in FF=self._FF(x) if you have it cached"""
        if FF is None:
            FF = self._FF(x, f)
        return FF @ x #[B,N,N]@[B,N,1]->[B,N,1]

    def _FF(self, x, f=None):
        """ diag(f(x)) - f(x)*f(x)^T
        Helper for computing Jacobian and partial. Pass in f=f(x) if you have it cached"""
        if f is None:
            f = self(x)
        #in-place helps with OOM
        FF = f.squeeze(-1).diag_embed() #[B,N,1] -> [B,N,N]
        FF -= f @ f.transpose(1, 2) #[B,N,N]-[B,N,1]@[B,1,N] -> [B,N,N]
        return FF


class Identity(Nonlinearity):
    def _function(self, x):
        """f_i = x_i"""
        return x #[B,M,1] or [M,1]

    def L(self, x):
        """L = 1/2 x^T*x --> dL/dx_i = x_i"""
        xx = x.transpose(-2,-1) @ x #[B,M,1]->[B,1,1] or #[M,1]->[1,1]
        return 0.5*xx.squeeze(-2) #[B,1,1]->[B,1] or #[1,1]->[1]

    def J(self, x, **kwargs):
        return eye_like(x)


class Spherical(Nonlinearity):
    def __init__(self, centering=False):
        """If centering is True, equivalent to LayerNorm (see Tang and Kopp 2021)"""
        super().__init__()
        self.centering = centering

    def __str__(self):
        return f'Spherical(centering={self.centering})'

    def _function(self, x, Z=None):
        """f_i = (x_i - m)/||x-m||, where m=mean(x) if centering==True, else 0"""
        if self.centering:
            x = x - x.mean(dim=-2, keepdim=True) #[B,M,1] or [M,1]
        if Z is None:
            Z = self._norm(x)
        return  x / Z

    def L(self, x):
        """f(x)_i = dL/dx_i"""
        return self._norm(x) #[B,M,1]->[B,1,1] or #[M,1]->[1,1]

    def J(self, x, f=None, Z=None):
        if self.centering:
            raise NotImplementedError()
        if Z is None:
            Z = self._norm(x)
        if f is None:
            f = self(x, Z)
        return (eye_like(f) - f@f.transpose(-2,-1))/Z

    def _norm(self, x):
        """||x|| = sqrt((x**2).sum(dim=-2)"""
        return torch.linalg.vector_norm(x, dim=-2, keepdim=True)


###########
# Helpers #
###########
def eye_like(x, **kwargs):
    """x is either [M,1] or [B,M,1]"""
    eye = torch.eye(x.shape[-2], device=x.device, **kwargs) #[M,M]
    if len(x.shape)==3: #x is [B,M,1]
        eye = eye.expand(x.shape[0], -1, -1) #[B,M,M]
    return eye


############
# Networks #
############
class LargeAssociativeMem(nn.Module):
    """
    Krotov and Hopfield 2021
    Setting f=Softmax(), g=Identity(), and dt=tau, reduces to Ramsauer et al. 2020
    """
    def __init__(self, input_size, hidden_size, f=Softmax(beta=1, train=False), g=Identity(), tau=1.,
                 normalize_weight=False, dropout=False, normalize_input=False, input_mode='init',
                 max_num_steps=float('inf'), dt=0.1, check_converged=True, fp_thres=0.001, fp_mode='iter',
                 **kwargs):
        if kwargs:
            #kwargs ignores any extra values passed in (eg. via a Config object)
            warnings.warn(f'Ignoring unused LargeAssociativeMem kwargs: {kwargs}')
        self._check_parameters(normalize_weight, dropout, normalize_input, input_mode, dt, fp_mode)

        super().__init__()
        self.M = input_size
        self.N = hidden_size

        self.g = g #input nonlin
        self.f = f #hidden nonlin

        self._W = nn.Parameter(torch.zeros(self.N, self.M))
        nn.init.xavier_normal_(self._W)
        self.normalize_weight = normalize_weight
        self._maybe_normalize_weight()

        self.dropout = dropout #only applies if net in in 'training' mode

        self.tau = torch.tensor(float(tau))
        self.dt = float(dt)
        self.max_num_steps = max_num_steps
        self.check_converged = check_converged
        self.fp_thres = float(fp_thres)
        self.fp_mode = fp_mode

        self.normalize_input = normalize_input
        self.input_mode = input_mode

        #hack: store final v before output g(v) during forward b/c needed for Jg
        self.v_last = None


    def _check_parameters(self, normalize_weight, dropout, normalize_input,
                          input_mode, dt, fp_mode):
        assert normalize_weight in [True, False]
        assert 0 < dropout < 1 or dropout is False
        assert normalize_input in [True, False]
        assert dt<=1, 'Step size dt should be <=1'

        assert fp_mode in ['iter', 'del2'], f"Invalid fixed point mode: '{fp_mode}'"
        if fp_mode == 'del2':
            raise NotImplementedError()

        input_modes = ['init', 'cont', 'clamp'] #valid modes are all combinations of these
        valid_input_modes = chain.from_iterable(combinations(input_modes, k) for k in range(1,len(input_modes)+1))
        valid_input_modes = ['+'.join(m) for m in valid_input_modes]
        assert input_mode in valid_input_modes, f"Invalid input mode: '{input_mode}'"


    def _maybe_normalize_weight(self):
        if self.normalize_weight:
            #W is normalized, _W is raw; computation is done with W, learning is done on _W
            self.W = self._W / self._W.norm(dim=1, keepdim=True) #[N,M]
        else:
            self.W = self._W


    def _parse_input(self, input):
        clamp_mask = None
        if type(input) == tuple:
            assert len(input) == 2
            input, clamp_mask = input

        if self.normalize_input:
            input /= input.norm(dim=-2, keepdim=True)

        v = torch.randn_like(input)/100
        external_current = torch.zeros_like(input)
        clamp_values = None
        if 'init' in self.input_mode:
            v = input
        if 'cont' in self.input_mode:
            external_current = input
        if 'clamp' in self.input_mode:
            assert clamp_mask is not None
            clamp_values = input[clamp_mask]
            v[clamp_mask] = clamp_values

        if self.training and self.dropout:
            batch_size, input_size, _ = input.shape #[B,M,1]
            self.dropout_mask = torch.rand(batch_size, self.hidden_size, 1) < self.dropout #[B,N,1]
            self.dropout_mask = self.dropout_mask.to(next(self.parameters()).device)

        return v, external_current, clamp_mask, clamp_values


    def _maybe_dropout(self, f):
        if self.training and self.dropout:
            f = f * ~self.dropout_mask
        return f


    def forward(self, input, debug=False):
        """input is either a [B,N,1]-dim tensor or a tuple (input, clamp_mask) where clamp_mask is
        a [B,N,1]-dim boolean tensor indicating which input units to clamp. clamp_value is inferred
        from input and clamp_mask as input[clamp_mask]. clamp_mask is ignored unless self.input_mode
        contains 'clamp'"""
        self._maybe_normalize_weight()
        v, external_current, clamp_mask, clamp_values = self._parse_input(input) #[B,M,1],[B,M,1]

        if debug:
            full_state_history = []
        update_magnitude = float('inf')
        current_step = 0
        while update_magnitude > self.fp_thres and current_step < self.max_num_steps:
            v_prev = v
            v = self.step(v_prev, external_current, clamp_mask, clamp_values, debug=debug) #[B,M,1]
            current_step += 1

            if debug: #v is actually full_state dict (see step() method)
                full_state_history.append(v) #store it
                v = v['v_next'] #extract the actual v from the dict

            with torch.no_grad():
                update_magnitude = (v_prev-v).norm()
                if self.check_converged and current_step >= self.max_num_steps and update_magnitude > self.fp_thres:
                    warnings.warn('Not converged: '
                          f'(update={update_magnitude}, fp_thres={self.fp_thres})')

        if debug:
            return full_state_history

        self.v_last = v
        return self.g(v)


    def step(self, v, I=torch.tensor(0.), clamp_mask=None, clamp_values=None, debug=False):
        """ v(t+dt) = v(t) + dt/tau_v [ -v(t) + W*f(h) ]

        Discretized version of continuous dynamics with tau_h --> 0 (i.e. h = Wg)
            tau_h*dh/dt = -h + W*g(v)
            tau_v*dv/dt = -v + W^T*f(h) + I
        """
        h = self.compute_hidden(self.g(v))
        f = self._maybe_dropout(self.f(h)) #[B,N,1]
        v_instant = self.compute_feature(f, I)
        v_next = v + (self.dt/self.tau)*(-v + v_instant) #[B,M,1]

        if 'clamp' in self.input_mode: #overwrite any updates to clamped units
            v_next[clamp_mask] = clamp_values

        if debug:
            full_state = {}
            for key in ['v', 'I', 'h', 'f', 'v_instant', 'v_next']:
                #TODO: should be storing v_next *before* update? (only matters for 'cont' input_mode)
                full_state[key] = locals()[key].detach()
            return full_state
        return v_next


    def compute_hidden(self, g):
        """h_u = \sum_i W_ui*g_i, ie. h = W*g"""
        return self.W @ g #[N,M],[B,M,1]->[B,N,1]


    def compute_feature(self, f, I=torch.tensor(0.)):
        """v_i = \sum_u W_iu*f_u + I_i, ie. v = W^T*f + I """
        return self.W.t() @ f + I #[M,N],[B,N,1]->[B,M,1]


    @torch.no_grad()
    def energy(self, v, I=torch.tensor(0.)):
        g = self.g(v)
        h = self.compute_hidden(g)
        f = self.f(h)
        Lv = self.g.L(v)
        Lh = self.f.L(h)

        vg = ((v).transpose(-2,-1) @ g).squeeze(-2)
        hf = (h.transpose(-2,-1) @ f).squeeze(-2)
        fWg = (f.transpose(-2,-1) @ self.W @ g).squeeze(-2)
        E = (vg - Lv) + (hf - Lh) - fWg
        return E



class ConvolutionalMem(nn.Module):
    """Krotov 2021, sec 4.3"""
    def __init__(self, x_size, x_channels, y_channels=5, kernel_size=10, stride=1, z_size=50,
                 dt=0.1, fp_thres=1e-6, max_num_steps=5000, **kwargs):
        if kwargs:
            #kwargs ignores any extra values passed in (eg. via a Config object)
            warnings.warn(f'Ignoring unused ConvolutionalMem kwargs: {kwargs}')
        super().__init__()

        self.fp_thres = fp_thres
        self.max_num_steps = max_num_steps
        self.input_mode = 'init'

        self.x_channels = x_channels #Cx
        self.y_channels = y_channels #Cy
        self.kernel_size = kernel_size #K
        self.stride = stride #S
        self.x_size = x_size #L
        self.y_size = math.floor((x_size-kernel_size)/stride)+1 #M=floor((L-K)/S)+1
        self.z_size = z_size #N

        self.conv = nn.Conv2d(self.x_channels, self.y_channels,
                              self.kernel_size, self.stride, bias=False)
        self.convT = nn.ConvTranspose2d(self.y_channels, self.x_channels,
                                         self.kernel_size, self.stride, bias=False)
        self.convT.weight = self.conv.weight

        self.W = nn.Parameter(torch.empty(self.z_size, self.y_channels*self.y_size**2)) #[N,CyMM]
        nn.init.xavier_normal_(self.W)

        self.dt = dt
        self.tau_x = 1.
        self.tau_y = 0.2

        self.nonlin_y = nn.Softmax(dim=1) #y is [B,Cy,M,M]
        self.nonlin_z = nn.Softmax(dim=-2) #z is [B,N,1]
        self.beta_y = 3.
        self.beta_z = 7.



    def fc(self, fy):
        """fy is post-nonlinearity, shape=[B,M,M,Cy]"""
        return self.W @ fy.reshape(fy.shape[0],-1,1) #[N,CyMM]@[B,CyMM,1]->[B,N,1]


    def fcT(self, fz):
        """fz is post-nonlinearity, shape=[B,N,1]"""
        out = self.W.t() @ fz #[CyMM,N]@[B,N,1]->[B,CyMM,1]
        return out.reshape(fz.shape[0], self.y_channels, self.y_size, self.y_size) #[B,Cy,M,M]


    def forward(self, input, debug=False):
        x,_ = input
        y = torch.zeros(x.shape[0], self.y_channels, self.y_size, self.y_size, device=x.device)
        for t in range(self.max_num_steps):
            x_prev, y_prev = x, y
            x,y = self.step(x_prev, y_prev) #[B,Cx,L,L], [B,Cy,M,M]
            # with torch.no_grad():
            #     update_magnitude = (x_prev-x).norm()
            #     if update_magnitude < self.fp_thres:
            #         break
        # if t >= self.max_num_steps-1:
        #     warnings.warn('Not converged: '
        #               f'(update={update_magnitude}, fp_thres={self.fp_thres})')
        with torch.no_grad():
            update_magnitude = (x_prev-x).norm()
        print(f'steps={t}, final_update={update_magnitude}')
        return x


    def step(self, x, y):

        z = self.fc( self.nonlin_y(self.beta_y*y) ) #[B,N,1]

        y_instant = self.conv(x) + self.fcT( self.nonlin_z(self.beta_z*z) )
        y = y + (self.dt/self.tau_y)*(-y + y_instant)  #[B,Cy,M,M]

        x_instant = self.convT(self.nonlin_y(y)) #[B,Cx,L,L]
        x = x + (self.dt/self.tau_x)*(-x + x_instant)
        return x, y
