import math
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons


train_data, test_data = data.get_data(**data_config)

is_exception = [l.item() in net.exceptions for l in train_data.labels]
exceptions_idx = [idx for idx,is_ex in enumerate(is_exception) if is_ex]
input_ex, target_ex, perturb_mask_ex, label_ex = train_data[exceptions_idx]
input, target, perturb_mask, label = train_data[0:100]
num_ex = len(input_ex)
input[-num_ex:] = input_ex
target[-num_ex:] = target_ex


def generate_correlated(num_samples, dim=None, template=None, frac_same=0.5):
    if template is None:
        template = torch.rand(dim)
    if dim is None:
        dim = template.numel()

    data = template.tile(num_samples, 1)
    for x in data:
        redraw_idx = torch.rand_like(x) > math.sqrt(frac_same)
        x[redraw_idx] = torch.rand(redraw_idx.sum())

    return data


def unique_vectors(mat):
    mat = deepcopy(mat)
    for i, row_i in enumerate(mat):
        if row_i.isnan().any():
            continue
        for row_j in mat[i+1:]:
            if (row_i == row_j).all():
                row_j[:] = float('nan')
    return mat#[~mat[:,0].isnan(),:]


class Updater():
    def __init__(self, ax):
        self.inputs = input
        self.ax = ax

        plt.figure()
        self.beta_slider = Slider(ax=plt.axes([0.1, 0.15, 0.8, 0.03]),
                             label='beta',
                             valmin=0, valmax=50,
                             valinit=net.hidden.nonlin.beta.item())
        self.beta_slider.on_changed(self.update_beta)

        self.perturb_slider = Slider(ax=plt.axes([0.1, 0.05, 0.8, 0.03]),
                                label='pert',
                                valmin=0, valmax=1,
                                valinit=0)
        self.perturb_slider.on_changed(self.update_pert)

        self.weight_buttons = [CheckButtons(ax=plt.axes([0.1*i, 0.2, 0.1, 0.7]),
                                labels=range(i*10,(i+1)*10),
                                actives=[True]*10)
                   for i in range(10)]
        [b.on_clicked(self.toggle_weight) for b in self.weight_buttons]


    def update_beta(self, val):
        net.hidden.nonlin.beta.data = torch.tensor(self.beta_slider.val)
        self.ax[1,0].set_title(f'Output, beta={self.beta_slider.val:.2f}')
        self.draw()

    def update_pert(self, val):
        perturb_frac = self.perturb_slider.val
        train_data.set_perturb_frac_and_num(perturb_frac)
        self.inputs = torch.cat([train_data[i][0].unsqueeze(0) for i in range(len(train_data))])
        self.ax[0,0].set_title(f'Input, perturb={perturb_frac:.2f}')
        self.draw()

    def toggle_weight(self, label):
        mask = torch.tensor([b.get_status() for b in self.weight_buttons]).flatten().unsqueeze(1)
        net.fc.weight.data = mask*train_data.data
        self.draw()

    def draw(self):
        with torch.no_grad():
            fixed_points = net(self.inputs)[0]

        self.ax[0,0].images[0].set_data(train_data.batch_to_grid(self.inputs))
        self.ax[0,1].images[0].set_data(train_data.batch_to_grid(target))
        self.ax[1,0].images[0].set_data(train_data.batch_to_grid(fixed_points))
        self.ax[1,1].images[0].set_data(train_data.batch_to_grid(unique_vectors(fixed_points)))

        fig.canvas.draw_idle()


#%% Parameters sliders
with torch.no_grad():
    fixed_points = net(input)[0]
fig, ax = train_data.plot_batch(inputs=input,
                                targets=target,
                                outputs=fixed_points, ax_rows=2,
                                **{'Outputs (unique)' : unique_vectors(fixed_points)})
fig.tight_layout()
updater = Updater(ax)
