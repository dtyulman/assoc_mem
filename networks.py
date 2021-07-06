import warnings

import scipy.optimize as spo
import torch
import torch.nn.functional as F
from torch import nn


class ModernHopfield(nn.Module):
    """Ramsauer et al 2020"""
    def __init__(self, input_size, hidden_size, beta=None, tau=None,
                 input_mode='init', num_steps=None, dt=1, fp_thres=0.001, fp_mode='iter'):
        super().__init__()
        self.input_size = input_size # M
        self.hidden_size = hidden_size # N

        self.W = nn.Parameter(torch.zeros(self.hidden_size, self.input_size)) #[N,M]
        nn.init.xavier_normal_(self.W)

        self.beta = nn.Parameter(torch.tensor(beta or 1.), requires_grad=(beta is None))
        self.tau = nn.Parameter(torch.tensor(tau or 1.), requires_grad=(tau is None))

        assert dt<=1, 'Step size dt should be <=1'
        self.dt = dt
        self.num_steps = int(1/self.dt) if num_steps is None else int(num_steps)

        assert fp_mode in ['iter', 'del2'], f"Invalid fp mode: '{fp_mode}'"
        self.fp_mode = fp_mode
        self.fp_thres = fp_thres

        assert input_mode in ['init', 'cont', 'init+cont'], f"Invalid input mode: '{input_mode}' "#, 'clamp']
        self.input_mode = input_mode


    def forward(self, input, debug=False):
        if 'init' in self.input_mode:
            state = input #[B,M,1]
        else:
            state = torch.zeros_like(input)

        if 'cont' not in self.input_mode:
            input = 0

        if debug:
            state_debug_history = []
        update_magnitude = float('inf')
        step = 0
        while update_magnitude > self.fp_thres and step < self.num_steps:
            prev_state = state
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


    def update_state(self, v, I=0., debug=False):
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
            for key in ['h', 'f', 'v_instant', 'state']:
                state_debug[key] = locals()[key].detach()
            return state_debug
        return state


    def energy(self, v):
        with torch.no_grad():
            #more general:
            # E = ((v.transpose(-2,-1) @ g).squeeze(-2) - Lv
            #      + (h.transpose(-2,-1) @ f).squeeze(-2) - Lh
            #      - (f.transpose(-2,-1) @ self.W @ g).squeeze(-2))
            vv = v.transpose(-2,-1) @ v
            lse = torch.logsumexp(self.W@v, -2)
            return 0.5*vv.squeeze(-2) - lse


    def _h(self, g):
        """h_u = \sum_i W_ui*g_i, ie. h = Wg"""
        return self.W @ g #[N,M],[B,M,1]->[B,N,1]


    def _f(self, h):
        """f_u = f(h)_u = softmax(h)_u"""
        return F.softmax(self.beta * h, dim=-2) #[B,N,1] or [N,1]


    def _v(self, f, I=0):
        """v_i = \sum_u W_iu*f_u + I_i, ie. v = Wf + I """
        return self.W.t() @ f + I #[M,N],[B,N,1]->[B,M,1]


    def _g(self, v):
        """g_i = g(v)_i = v_i"""
        return v #[B,M,1] or [M,1]


    def _Lh(self, h):
        """f_u(h) = dLh/dh"""
        return torch.logsumexp(h, -2) #[B,N,1]->[B,1] or #[N,1]->[1]


    def _Lv(self, v):
        vv = v.transpose(-2,-1)@v #[B,M,1]->[B,1,1] or #[M,1]->[1,1]
        return 0.5*vv.squeeze(-2) #[B,1,1]->[B,1] or #[1,1]->[1]
