import torch
import torch.nn.functional as F
from torch import nn


class ModernHopfield(nn.Module):
    """Ramsauer et al 2020"""
    def __init__(self, input_size, hidden_size, beta=None, tau=None,
                 num_steps=None, dt=1, fp_threshold=0.001):
        super().__init__()
        self.input_size = input_size # M
        self.hidden_size = hidden_size # N

        self.mem = nn.Parameter(torch.zeros(self.hidden_size, self.input_size)) #[N,M]
        nn.init.xavier_normal_(self.mem)

        self.beta = nn.Parameter(torch.tensor(beta or 1.), requires_grad=(beta is None))
        self.tau = nn.Parameter(torch.tensor(tau or 1.), requires_grad=(tau is None))

        assert dt<=1, 'Step size dt should be <=1'
        self.dt = dt
        self.num_steps = int(1/self.dt) if num_steps is None else num_steps
        self.fp_threshold = fp_threshold


    def forward(self, input, debug=False):
        state = input #[B,M,1]
        if debug:
            state_debug_history = []
        update_magnitude = float('inf')
        step = 0
        while update_magnitude > self.fp_threshold and step < self.num_steps:
            prev_state = state
            state = self.update_state(prev_state, debug=debug) #[B,M,1]
            if debug: #state is actually state_debug
                state_debug_history.append(state) #store it
                state = state['state'] #extract the actual state from the dict
            with torch.no_grad():
                update_magnitude = (prev_state-state).norm()
            step += 1
        if debug:
            return state_debug_history
        return state


    def update_state(self, state, debug=False):
        """
        Continuous: tau*dv/dt = -v + X*softmax(X^T*v)
        Discretized: v(t+dt) = v(t) + dt/tau [-v(t) + X*softmax(X^T*v(t))]
        """
        h = torch.matmul(self.mem, state) #[N,M],[B,M,1]->[B,N,1]
        f = F.softmax(self.beta*h, dim=1) #[B,N,1]
        memT_f = torch.matmul(self.mem.t(), f) #[M,N],[B,N,1]->[B,M,1]
        state = state + (self.dt/self.tau)*(memT_f - state) #[B,M,1]
        if debug:
            state_debug = {}
            for key in ['h', 'f', 'memT_f', 'state']:
                state_debug[key] = locals()[key].detach()
            return state_debug
        return state
