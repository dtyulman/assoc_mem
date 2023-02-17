import itertools, os

from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from networks import LargeAssociativeMemoryWithCurrents, LargeAssociativeMemory
from data import AssociativeMNIST
import components as nc

#%%
class LargeAssociativeMemoryWithCurrents(LargeAssociativeMemory):
    def __init__(self, state_mode='currents', dynamics_mode='hidden_prev', *args, **kwargs):
        """
        if state_mode == 'currents':
            feature = v, hidden = h (total current into layer)

            if dynamics_mode == 'hidden_prev':
                Init:
                    1. v[0] init
                    2. h[0] = W*G(v[0])
                Loop:
                    3. v[t+1] = (1-dt/tau) v[t] + dt/tau W^T*F(h[t])
                              = (1-dt/tau) v[t] + dt/tau W^T*F(W*G(v[t])) b/c h[t+1] = W*G(v[t+1])
                    4. h[t+1] = W*G(v[t+1])

            elif dynamics_mode == 'instantaneous':
                Init:
                    1. v[0] init
                    1. h[0] arbitrary (never used)
                Loop:
                    3. h[t+1] = W*G(v[t])
                    4. v[t+1] = (1-dt/tau) v[t] + dt/tau W^T * F(h[t+1])
                              = (1-dt/tau) v[t] + dt/tau W^T * F(W*G(v[t])) b/c h[t+1] = W*G(v[t])
            Note v trajectory is the same in both cases, h trajectory is just offset by dt


        elif state_mode == 'rates':
            feature = g, hidden = f (firing rate of layer)

            if dynamics_mode == 'hidden_prev':
                Init:
                    1. g[0] init
                    2. f[0] = F(W*g[0]) because tau_f==0
                Loop:
                    3. g[t+1] = (1-dt/tau) g[t] + dt/tau G(W^T * f[t])
                              = (1-dt/tau) g[t] + dt/tau G(W^T * F(W*g[t])) b/c f[t+1] = F(W*g[t+1])
                    4. f[t+1] = F(W*g[t+1])

            elif dynamics_mode == 'instantaneous':
                Init:
                    1. g[0] init
                    1. f[0] arbitrary (never used)
                Loop:
                    3. f[t+1] = F(W*g[t])
                    4. g[t+1] = (1-dt/tau) g[t] + dt/tau G(W^T * f[t+1])
                              = (1-dt/tau) g[t] + dt/tau G(W^T * F(W*g[t])) b/c f[t+1] = F(W*g[t])
            Note g trajectory is the same in both cases, f trajectory is just offset by dt

            else: generally (if tau_f != 0)
                Init:
                    1. g[0] init
                    1. f[0] init
                Loop:
                    3. f[t+1] = (1-dt/tau) f[t] + dt/tau F(W*g[t])
                    3. g[t+1] = (1-dt/tau) g[t] + dt/tau G(W^T*f[t])

                      if used f[t+1] = F(W*g[t]) naively for tau=0
                    would get g[t+1] = (1-dt/tau) g[t] + dt/tau G(W^T* F(W*g[t-1]) ) WRONG!
                    should be g[t+1] = (1-dt/tau) g[t] + dt/tau G(W^T* F(W*g[t]) ) CORRECT for instantaneous f dynamics

        """

        assert state_mode in ['currents', 'rates']
        self.state_mode = state_mode
        assert dynamics_mode in ['hidden_prev', 'instantaneous']
        self.dynamics_mode = dynamics_mode

        super().__init__(*args, **kwargs)


    def forward(self, *init_state, clamp_mask=None, return_mode=''):
        output = super().forward(*init_state, clamp_mask=clamp_mask, return_mode=return_mode)
        if return_mode == 'trajectory':
            return output
        elif return_mode == '':
            feature, hidden = output
            if self.state_mode == 'currents':
                feature = self.feature.nonlin(feature)
                hidden = self.hidden.nonlin(hidden)
            return feature, hidden
        else:
            raise ValueError()


    def default_init(self, feature_init, hidden_init=None):
        if self.dynamics_mode == 'hidden_prev':
            if self.state_mode == 'rates':
                hidden_init = self.hidden.nonlin(self.fc(feature_init))
            elif self.state_mode == 'currents':
                hidden_init = self.fc(self.feature.nonlin(feature_init))
            return feature_init, hidden_init
        return super().default_init(feature_init, hidden_init)


    def step(self, feature_prev, hidden_prev):
        if self.state_mode == 'rates':
            g_prev, f_prev = feature_prev, hidden_prev
            if self.dynamics_mode == 'instantaneous':
                f = self.hidden.nonlin(self.fc(g_prev))
                g_ss = self.feature.nonlin(self.fc.T(f))
                g = (1-self.feature.eta)*g_prev + self.feature.eta*g_ss

            elif self.dynamics_mode == 'hidden_prev':
                g_ss = self.feature.nonlin(self.fc.T(f_prev))
                g = (1-self.feature.eta)*g_prev + self.feature.eta*g_ss
                f = self.hidden.nonlin(self.fc(g))
            return g, f

        elif self.state_mode == 'currents':
            v_prev, h_prev = feature_prev, hidden_prev
            if self.dynamics_mode == 'instantaneous':
                h = self.fc(self.feature.nonlin(v_prev))
                v_ss = self.fc.T(self.hidden.nonlin(h))
                v = (1-self.feature.eta)*v_prev + self.feature.eta*v_ss

            elif self.dynamics_mode == 'hidden_prev':
                v_ss = self.fc.T(self.hidden.nonlin(h_prev))
                v = (1-self.feature.eta)*v_prev + self.feature.eta*v_ss
                h = self.fc(self.feature.nonlin(v))
            return v, h


    def energy(self, feature, hidden, debug=False):
        if self.state_mode == 'rates':
            g, f = feature, hidden
            return super().energy(g, f, debug=debug)
        elif self.state_mode == 'currents':
            #E = [vTg - Lv] + [hTf - Lh] - fTWg #from KH2020
            v, h = feature, hidden

            #with exception of the change-of-vars this is the same as the 'rates' energy
            G = self.feature.nonlin(v) #[T,B,N]
            F = self.hidden.nonlin(h) #[T,B,M]

            if isinstance(self.feature.nonlin, nc.Spherical):
                E_feature = torch.zeros(v.shape[:-1])
            elif isinstance(self.feature.nonlin, nc.Identity):
                #[T,B,1,N]@T,B,N,1]->[T,B,1,1]
                Lv = 0.5*(v.unsqueeze(-2)@v.unsqueeze(-1)).squeeze() #[T,B]
            else:
                raise NotImplementedError()
            if not isinstance(self.feature.nonlin, nc.Spherical):
                E_feature = (v*G).sum(dim=-1) - Lv #[T,B]

            if isinstance(self.hidden.nonlin, nc.Softmax):
                beta = self.hidden.nonlin.beta
                Lh = torch.logsumexp(beta*h, dim=-1)/beta #[T,B]
            elif isinstance(self.hidden.nonlin, nc.Polynomial):
                n = self.hidden.nonlin.n
                Lh = torch.pow(h,n+1).sum(dim=-1) #[T,B]
            else:
                raise NotImplementedError()
            E_hidden = (h*F).sum(dim=-1) - Lh #[T,B]

            #[T,B,1,M]@[M,N]@[T,B,N,1]->[T,B,1,1]->[T,B]
            E_syn = -(F.unsqueeze(-2) @ self.fc.weight @ G.unsqueeze(-1)).squeeze()

            E = E_feature + E_hidden + E_syn #[T,B]
            if debug:
                return E_feature, E_hidden, E_syn, E
            return E



#%%
# net = networks.LargeAssociativeMemoryWithCurrents(input_size=net.feature.shape[0],
#                                                   hidden_size=net.hidden.shape[0],
#                                                   input_nonlin = net.feature.nonlin,
#                                                   hidden_nonlin = net.hidden.nonlin,
#                                                   input_mode = net.input_mode,
#                                                   tau = net.feature.tau,
#                                                   dt = net.dt,
#                                                   max_steps = net.max_steps,
#                                                   state_mode='rates',
#                                                   dynamics_mode='hidden_prev')
# net.fc._weight.data = _net.fc.weight.clone()
# net.max_steps = 100
# with torch.no_grad():
#     state_trajectory = net(input, clamp_mask=~perturb_mask)

#%%
root = '/Users/danil/My/School/Columbia/Research/assoc_mem/results/cluster_mountpoint/2023-01-12/LargeAssociativeMemory_EnergyDebug_0000'

train_data = AssociativeMNIST(perturb_entries=0.5, perturb_mask='rand', perturb_value='min',
                              n_per_class=10, normalize=True)
train_loader = DataLoader(train_data, batch_size=2, shuffle=False)
batch = next(iter(train_loader))

# batch[2][:] = True #unclamp all: set perturb_mask=True st. clamp_mask=~perturb_mask=False

#%%
beta_list = [1]#, 10]
state_mode_list = ['rates']#, 'currents']
dynamics_mode_list = ['instantaneous']#, 'hidden_prev']

for beta, state_mode, dynamics_mode in itertools.product(beta_list, state_mode_list, dynamics_mode_list):
    cfg_label = f'hidden_nonlin=Softmax(beta={beta}) state_mode={state_mode} dynamics_mode={dynamics_mode}'
    path = os.path.join(root, cfg_label, 'checkpoints')
    path = os.path.join(path, next(os.walk(path))[-1][0])

    # net = LargeAssociativeMemoryWithCurrents.load_from_checkpoint(path)
    # net.hidden.nonlin = Polynomial(2)
    # net.feature.nonlin = Identity()

    net = LargeAssociativeMemoryWithCurrents(input_size=784,
                                            hidden_size=100,
                                            input_nonlin = nc.Identity(),
                                            hidden_nonlin = nc.Polynomial(2),
                                            input_mode = 'clamp',
                                            tau = 1,
                                            dt = 0.05,
                                            max_steps = 2000,
                                            converged_thres=1e-6,
                                            state_mode=state_mode,
                                            dynamics_mode=dynamics_mode)

    with torch.no_grad():
        E = net.get_energy_trajectory(batch)
        dE = torch.diff(E, dim=0)

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(E)
    ax[1].plot(dE)

    ax[0].set_title(f'Id()/Poly(2) {state_mode} {dynamics_mode}')
    ax[0].set_ylabel('$E$')
    ax[1].axhline(0, color='k', lw=0.3)
    ax[1].set_ylabel('$\Delta E$')
    ax[1].set_xlabel('Time')
    ax[1].set_ylim(bottom=-3e-6, top=dE[~dE.isnan()].max().item())
    [l.set_linewidth(0.5) for a in ax.flatten() for l in a.get_lines()]

    fig.set_size_inches(3.5,5)
    fig.tight_layout()

    # fig.savefig(f'./results/2023-01-13/energy_debug beta={beta} state_mode={state_mode} dynamics_mode={dynamics_mode} unclamped.png',
    #             pad_inches=0, transparent=True)
