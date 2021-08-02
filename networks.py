import warnings
from itertools import chain, combinations

import torch
import torch.nn.functional as F
from torch import nn


class ModernHopfield(nn.Module):
    """Ramsauer et al 2020"""
    def __init__(self, input_size, hidden_size, beta=None, tau=None, normalize_weight=False,
                 dropout=False, normalize_input=False, input_mode='init', num_steps=None, dt=1,
                 fp_thres=0.001, fp_mode='iter', **kwargs):
        if kwargs:
            #kwargs ignores any extra values passed in (eg. via a Config object)
            warnings.warn(f'Ignoring unused ModernHopfield kwargs: {kwargs}')

        super().__init__()
        self.input_size = input_size # M
        self.hidden_size = hidden_size # N

        self._W = nn.Parameter(torch.zeros(self.hidden_size, self.input_size)) #[N,M]
        nn.init.xavier_normal_(self._W)
        assert normalize_weight in [True, False]
        self.normalize_weight = normalize_weight
        self._maybe_normalize_weight()
        assert 0 < dropout < 1 or dropout is False
        self.dropout = dropout #only applies if net in in 'training' mode

        self.beta = nn.Parameter(torch.tensor(beta or 1.), requires_grad=(beta is None))
        self.tau = nn.Parameter(torch.tensor(tau or 1.), requires_grad=(tau is None))

        assert dt<=1, 'Step size dt should be <=1'
        self.dt = dt
        self.num_steps = int(1/self.dt) if num_steps is None else int(num_steps)

        assert fp_mode in ['iter'], f"Invalid fixed point mode: '{fp_mode}'" #TODO: 'del2'
        self.fp_mode = fp_mode
        self.fp_thres = fp_thres

        self.normalize_input = normalize_input
        self.input_mode = input_mode.lower()
        input_modes = ['init', 'cont', 'clamp'] #valid modes are all combinations of these
        valid_input_modes = chain.from_iterable(combinations(input_modes, k) for k in range(1,len(input_modes)+1))
        valid_input_modes = ['+'.join(m) for m in valid_input_modes]
        assert self.input_mode in valid_input_modes, f"Invalid input mode: '{input_mode}'"


    def _maybe_normalize_weight(self):
        if self.normalize_weight:
            #W is normalized, _W is raw; computation is done with W, learning is done on _W
            self.W = self._W / self._W.norm(dim=1, keepdim=True)
        else:
            self.W = self._W


    def _initialize_input(self, input, clamp_mask):
        if self.normalize_input:
            input /= input.norm(dim=1, keepdim=True)

        state = torch.zeros_like(input)
        external_current = torch.zeros_like(input)
        clamp_values = None
        if 'init' in self.input_mode:
            state = input
        if 'cont' in self.input_mode:
            external_current = input
        if 'clamp' in self.input_mode:
            clamp_values = input[clamp_mask]

        if self.training and self.dropout:
            batch_size, input_size, _ = input.shape #[B,M,1]
            self.dropout_mask = torch.rand(batch_size, self.hidden_size, 1) < self.dropout #[B.N,1]

        return state, external_current, clamp_values


    def _maybe_dropout(self, f):
        if self.training and self.dropout:
            f[self.dropout_mask] = 0
        return f


    def forward(self, input, clamp_mask=None, debug=False):
        self._maybe_normalize_weight()
        state, external_current, clamp_values = self._initialize_input(input, clamp_mask) #[B,M,1],[B,M,1]

        if debug:
            state_debug_history = []
        update_magnitude = float('inf')
        step = 0
        while update_magnitude > self.fp_thres and step < self.num_steps:
            prev_state = state
            state = self.update_state(prev_state, external_current, clamp_mask, clamp_values, debug=debug) #[B,M,1]
            step += 1

            if debug: #state is actually state_debug
                state_debug_history.append(state) #store it
                state = state['state'] #extract the actual state from the dict

            #check convergence
            with torch.no_grad():
                update_magnitude = (prev_state-state).norm()
            if step > self.num_steps and update_magnitude > self.fp_thres:
                warnings.warn('Not converged: '
                      f'(update={update_magnitude}, fp_thres={self.fp_thres})')

        if debug:
            return state_debug_history
        return state

    # def forward(self, input, debug=False):
    #     if debug:
    #         assert self.fp_mode == 'iter', "Debug logging not implemented for fp_mode=='del2'"
    #         state_debug_history = []

    #     p0 = input #[B,M,1]
    #     for step in range(self.num_steps):
    #         p1 = self.update_state(p0, debug=debug) #[B,M,1]
    #         if self.fp_mode=='del2': #like scipy.optimize.fixed_point() w/ use_accel=True
    #https://github.com/scipy/scipy/blob/v1.7.0/scipy/optimize/minpack.py#L899-L942
    #             p2 = self.update_state(p1)
    #             d = p2 - 2.*p1 + p0
    #             del2 = p0 - torch.square(p1-p0)/d
    #             p = torch.where(d!=0, del2, p2) #TODO:don't evaluate del2 where d!=0
    #         else:
    #             p = p1
    #         if debug: #state is actually state_debug
    #             state_debug_history.append(p) #store it
    #             p = p['state'] #extract the actual state from the dict
    #         update_magnitude = (p0-p).norm()
    #         if update_magnitude < self.fp_thres:
    #             break
    #         p0 = p

    #     if step >= self.num_steps:
    #         print('Warning: reached max num steps without convergence: '
    #               f'(update={update_magnitude}, fp_thres={self.fp_thres})')
    #     if debug:
    #         return state_debug_history
    #     return p


    def update_state(self, v, I=torch.tensor(0.), clamp_mask=None, clamp_values=None, debug=False):
        """
        Continuous: tau*dv/dt = -v + X*softmax(X^T*v)
        Discretized: v(t+dt) = v(t) + dt/tau [-v(t) + X*softmax(X^T*v(t))]
        """
        h = self.W @ self._g(v) #[N,M],[B,M,1]->[B,N,1]
        f = self._maybe_dropout(self._f(h)) #[B,N,1]
        v_instant = self.W.t() @ f + I #[M,N],[B,N,1]->[B,M,1]
        state = v + (self.dt/self.tau)*(v_instant - v) #[B,M,1]
        #     = (1-dt/tau)*v + (dt/tau)*v_instant

        if self.input_mode == 'clamp':
            state[clamp_mask] = clamp_values

        if debug:
            state_debug = {}
            for key in ['I', 'h', 'f', 'v_instant', 'state', 'v']:
                #TODO: should be storing state *before* update? (only matters for 'cont' input_mode)
                state_debug[key] = locals()[key].detach()
            return state_debug
        return state


    @torch.no_grad()
    def energy(self, v, I=torch.tensor(0.)):
        #general:
        # g = self._g(v)
        # h = self._h(g)
        # f = self._f(h)
        # Lv = self._Lv(v)
        # Lh = self._Lh(h)

        # vg = ((v).transpose(-2,-1) @ g).squeeze(-2)
        # hf = (h.transpose(-2,-1) @ f).squeeze(-2)
        # fWg = (f.transpose(-2,-1) @ self.W @ g).squeeze(-2)
        # E = (vg - Lv) + (hf - Lh) - fWg

        #assuming f=softmax, g=id
        vv = ((v/2. - I).transpose(-2,-1) @ v).squeeze(-2)
        lse = torch.logsumexp(self.beta*self.W@v, -2)/self.beta
        E = vv - lse
        return E


    def _h(self, g):
        """h_u = \sum_i W_ui*g_i, ie. h = W*g"""
        return self.W @ g #[N,M],[B,M,1]->[B,N,1]


    def _f(self, h):
        """f_u = f(h)_u = softmax(h)_u"""
        return F.softmax(self.beta * h, dim=-2) #[B,N,1] or [N,1]


    def _v(self, f, I=torch.tensor(0.)):
        """v_i = \sum_u W_iu*f_u + I_i, ie. v = W^T*f + I """
        return self.W.t() @ f + I #[M,N],[B,N,1]->[B,M,1]


    def _g(self, v):
        """g_i = g(v)_i = v_i"""
        return v #[B,M,1] or [M,1]


    def _Lh(self, h):
        """f_u(h) = dLh/dh_u"""
        return torch.logsumexp(self.beta*h, -2)/self.beta #[B,N,1]->[B,1] or #[N,1]->[1]


    def _Lv(self, v):
        """g_i(v) = dLv/dv_i"""
        vv = v.transpose(-2,-1) @ v #[B,M,1]->[B,1,1] or #[M,1]->[1,1]
        return 0.5*vv.squeeze(-2) #[B,1,1]->[B,1] or #[1,1]->[1]
