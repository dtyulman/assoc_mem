import warnings

import torch
import torch.nn.functional as F
from torch import nn


class ModernHopfield(nn.Module):
    """Ramsauer et al 2020"""
    def __init__(self, input_size, hidden_size, beta=None, tau=None, normalize_weight=False,
                 normalize_input=False, input_mode='init', num_steps=None, dt=1,
                 fp_thres=0.001, fp_mode='iter'):

        super().__init__()
        self.input_size = input_size # M
        self.hidden_size = hidden_size # N

        self._W = nn.Parameter(torch.zeros(self.hidden_size, self.input_size)) #[N,M]
        nn.init.kaiming_normal_(self._W)
        assert normalize_weight in [True, False]
        self.normalize_weight = normalize_weight
        self._maybe_normalize_weight()

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
        assert self.input_mode in ['init', 'cont', 'init+cont', 'clamp'], f"Invalid input mode: '{input_mode}'"
        if self.input_mode == 'clamp':
            self.clamp_mask = torch.zeros(self.input_size, 1) #[M,1]
            self.clamp_values = None


    def _maybe_normalize_weight(self):
        if self.normalize_weight:
            #W is normalized, _W is raw; computation is done with W, learning is done on _W
            self.W = self._W / self._W.norm(dim=1, keepdim=True)
        else:
            self.W = self._W


    def _set_state_and_input(self, input):
        if self.normalize_input:
            input /= input.norm(dim=1, keepdim=True)

        if self.input_mode == 'init':
            state = input
            input = torch.zeros_like(input)
        elif self.input_mode == 'cont':
            state = torch.zeros_like(input)
        elif self.input_mode == 'init+cont':
            state = input
        elif self.input_mode == 'clamp':
            state = torch.zeros_like(input)
            input = torch.zeros_like(input)
            self.clamp_values = input
            self.clamp_mask = self.clamp_mask.expand_as(input) #[M,1]->[B,M,1] or no-op

        return state, input


    def _maybe_clamp(self, state):
        """ Runs at every step of the dynamics. Overwrites the clamped neurons, specified
        by boolean clamp_mask"""
        if self.input_mode == 'clamp':
            state[self.clamp_mask] = self.clamp_values[self.clamp_mask]
        return state


    def forward(self, input, debug=False):
        self._maybe_normalize_weight()
        state, input = self._set_state_and_input(input) #[B,M,1],[B,M,1]

        if debug:
            state_debug_history = []
        update_magnitude = float('inf')
        step = 0
        while update_magnitude > self.fp_thres and step < self.num_steps:
            prev_state = self._maybe_clamp(state)
            state = self.update_state(prev_state, input, debug=debug) #[B,M,1]
            step += 1

            if debug: #state is actually state_debug
                state_debug_history.append(state) #store it
                state = state['state'] #extract the actual state from the dict
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


    def update_state(self, v, I=torch.tensor(0.), debug=False):
        """
        Continuous: tau*dv/dt = -v + X*softmax(X^T*v)
        Discretized: v(t+dt) = v(t) + dt/tau [-v(t) + X*softmax(X^T*v(t))]
        """
        h = self.W @ self._g(v) #[N,M],[B,M,1]->[B,N,1]
        f = self._f(h) #[B,N,1]
        v_instant = self.W.t() @ f + I #[M,N],[B,N,1]->[B,M,1]
        state = v + (self.dt/self.tau)*(v_instant - v) #[B,M,1]
        if debug:
            state_debug = {}
            for key in ['I', 'h', 'f', 'v_instant', 'state']:
                state_debug[key] = locals()[key].detach()
            return state_debug
        return state


    @torch.no_grad()
    def energy(self, v, I=torch.tensor(0.)):
        #more general:
        # g = self._g(v)
        # h = self._h(g)
        # f = self._f(h)
        # Lv = self._Lv(v)
        # Lh = self._Lh(h)

        # vg = ((v-I).transpose(-2,-1) @ g).squeeze(-2)
        # hf = (h.transpose(-2,-1) @ f).squeeze(-2)
        # fWg = (f.transpose(-2,-1) @ self.W @ g).squeeze(-2)
        # E = (vg - Lv) + (hf - Lh) - fWg

        vv = ((v/2. - I).transpose(-2,-1) @ v).squeeze(-2)
        lse = torch.logsumexp(self.W@v, -2)
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
        """f_u(h) = dLh/dh"""
        return torch.logsumexp(h, -2) #[B,N,1]->[B,1] or #[N,1]->[1]


    def _Lv(self, v):
        """g_i(v) = dLv/dv"""
        vv = v.transpose(-2,-1) @ v #[B,M,1]->[B,1,1] or #[M,1]->[1,1]
        return 0.5*vv.squeeze(-2) #[B,1,1]->[B,1] or #[1,1]->[1]
