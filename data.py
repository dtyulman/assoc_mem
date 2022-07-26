from copy import deepcopy

import numpy as np
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import plots, networks

class AssociativeDataset(Dataset):
    def __init__(self, data, labels=None, perturb_mask='last', perturb_entries=0.5, perturb_value=0):
        self.data = data
        self.labels = labels

        self.set_perturb_frac_and_num(perturb_entries)
        self.set_perturb_value(perturb_value)
        self.set_perturb_mask(perturb_mask)


    def set_perturb_frac_and_num(self, perturb_entries):
        total_entries = self.data[0].numel()
        if perturb_entries < 1: #specified the fraction of entries to perturb
            self.perturb_frac = perturb_entries
            self.perturb_num = int(total_entries*self.perturb_frac)
        else: #specified the number of entries to perturb
            self.perturb_num = perturb_entries
            self.perturb_frac = float(self.perturb_num) / total_entries
        assert 0 < self.perturb_frac < 1, 'Must have 0 < perturb_frac ({self.perturb_frac}) < 1'
        assert self.perturb_num < total_entries, 'Must have perturb_num ({self.perturb_num}) < input_size ({total_entries})'


    def set_perturb_value(self, value):
        """Sets self._perturb_value to either a string for dynamic generation
        of the value, or a fixed number
        """
        assert np.isreal(value) or value in ['rand', 'flip', 'min', 'max']
        if value == 'min':
            self._perturb_value = self.data.min()
        elif value == 'max':
            self._perturb_value = self.data.max()
        else: #'rand', 'flip', or <number>
            self._perturb_value = value


    def get_perturb_value(self, datapoint, mask):
        if self._perturb_value == 'rand':
            return torch.rand(mask.numel())
        elif self._perturb_value == 'flip': #TODO: implement for 0/1 data
            return -datapoint[mask]
        return self._perturb_value


    def set_perturb_mask(self, mask):
        """Sets self._perturb_mask to either None for random generation of the
        mask for each datapoint, or a fixed matrix same shape as one of the datapoints.

        Override if datapoint is not a vector (e.g. 3D image tensor)
        """
        assert mask in ['first', 'last', 'rand']
        datapoint = self.data[0]
        if mask == 'first':
            self._perturb_mask = torch.zeros_like(datapoint, dtype=bool)
            self._perturb_mask[:self.perturb_num] = 1
        elif mask == 'last':
            self._perturb_mask = torch.zeros_like(datapoint, dtype=bool)
            self._perturb_mask[-self.perturb_num:] = 1
        elif mask == 'rand':
            self._perturb_mask = None #generated per datapoint
        else:
            raise ValueError('Invalid perturb mask')


    def get_perturb_mask(self, datapoint):
        if self._perturb_mask is None:
            return torch.rand_like(datapoint) < self.perturb_frac
        return self._perturb_mask


    def perturb(self, datapoint):
        datapoint = deepcopy(datapoint) #prevent in-place modification
        mask = self.get_perturb_mask(datapoint)
        value = self.get_perturb_value(datapoint, mask)
        datapoint[mask] = value
        return datapoint, mask


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        target = self.data[index]
        input, perturb_mask = self.perturb(target)
        return input, target, perturb_mask


    @staticmethod
    def batch_to_grid(batch):
        """batch is [B,N],
        returns sqrt(B)-by-sqrt(B) grid of sqrt(N)-by-sqrt(N) images

        Override if datapoint is not a vector (e.g. 3D image tensor)"""
        imgs = plots.rows_to_images(batch)
        grid = plots.images_to_grid(imgs, vpad=1, hpad=1)
        return grid


    def plot_batch(self, inputs=None, targets=None, outputs=None, num_samples=100):
        if inputs is None and targets is None:
            inputs, targets, _ = next(iter(
                DataLoader(self, batch_size=num_samples, shuffle=True)
                ))

        named_batches = {'Inputs': inputs, 'Targets': targets, 'Outputs': outputs}
        grids, titles = [], []
        for title, batch in named_batches.items():
            if batch is not None:
                grids.append(self.batch_to_grid(batch.cpu()))
                titles.append(title)

        return plots.plot_matrices(grids, titles, ax_rows=1)



class AssociativeRandom(AssociativeDataset):
    def __init__(self, num_samples=50, input_size=20, num_classes=2, normalize=False,
                 data_kwargs={'distribution':'bern'},
                 **perturb_kwargs):

        distribution = data_kwargs.pop('distribution', 'bern')
        if distribution == 'bern':
            p = data_kwargs.pop('p', 0.5)
            balanced = data_kwargs.pop('balanced', False)
            data = (torch.rand(num_samples, input_size, 1) > p).to(torch.get_default_dtype())
            if balanced:
                data = 2*data - 1
        elif distribution == 'gaus':
            mean = data_kwargs.pop('mean', 0)
            std = data_kwargs.pop('std', 1)
            data = torch.randn(num_samples, input_size, 1)*std + mean
        elif distribution == 'unif':
            lo = data_kwargs.pop('lo', 0)
            hi = data_kwargs.pop('hi', 1)
            data = torch.rand(num_samples, input_size, 1)*(hi-lo) + lo
        else:
            raise ValueError(f"Invalid distribution spec: '{distribution}'")

        labels = torch.randint(num_classes,(num_samples,))

        super().__init__(data, labels, **perturb_kwargs)



class AssociativeMNIST(AssociativeDataset):
    def __init__(self,
                 num_samples=None,
                 select_classes='all',
                 n_per_class='all',
                 train_or_test='train',
                 crop=False,
                 downsample=False,
                 normalize=False,
                 **perturb_kwargs):
        data, labels = self.load_and_preprocess(num_samples, select_classes, n_per_class,
                                                train_or_test, crop, downsample, normalize)
        super().__init__(data, labels, **perturb_kwargs)


    def load_and_preprocess(self, num_samples, select_classes, n_per_class,
                            train_or_test, crop, downsample, normalize):
        #load
        mnist = torchvision.datasets.MNIST(root='./data/', download=True, train=(train_or_test=='train'))
        data = mnist.data #shape=[D,28,28], pixels in range [0-255]
        labels = mnist.targets #shape=[D], entries in [0-9]

        #get subset
        data, labels = filter_by_class(data, labels, num_samples, select_classes, n_per_class)

        #preprocess data
        data = (data-data.min())/(data.max()-data.min()) #to range [0,1]
        if crop: #remove `crop` pixels from top,bottom,left,right
            data = data[:, crop:-crop, crop:-crop]
        if downsample: #take every `downsample`th pixel vertically and horizontally
            data = data[:, ::downsample, ::downsample]
        data = data.reshape(data.shape[0], -1) #flatten
        if normalize: #normalize each row to unit vector
            data = data/data.norm(dim=1, keepdim=True)

        return data, labels



def AssociativeClassifyMNIST(AssociativeMNIST):
    def __init__(self,
                 num_samples=None,
                 select_classes='all',
                 n_per_class='all',
                 train_or_test='train',
                 crop=False,
                 downsample=False,
                 normalize=False, #'image_only', 'full_input', False
                 perturb_value=0
                 ):

        assert normalize in ['image_only', 'full_input', False]
        data, labels = self.load_and_preprocess(num_samples, select_classes, n_per_class,
                                                train_or_test, crop, downsample,
                                                normalize=(normalize=='image_only'))

        #data is concatenated flattened_image + onehot_target
        num_classes = len(labels.unique())
        targets = F.one_hot(labels, num_classes=num_classes)
        data = torch.cat((data,targets))
        if normalize == 'full_input':
            data = data/data.norm(dim=1, keepdim=True)

        super().__init__(data, labels,
                         perturb_mask='last',
                         perturb_entries=num_classes,
                         perturb_value=perturb_value)



class AssociativeCIFAR10(AssociativeDataset):
    def __init__(self, num_samples=None, test=False, **perturb_kwargs):
        #get data
        cifar = torchvision.datasets.CIFAR10(root='./data/CIFAR10', download=True, train=(not test))
        data = torch.tensor(cifar.data) #in range [0, 255]
        labels = torch.tensor(cifar.targets)

        #get subset
        data, labels = filter_by_class(data, labels, num_samples=num_samples)

        data = (data-data.min())/(data.max()-data.min()) #to range [0., 1.]
        data = data.transpose(-1,-2).transpose(-2,-3) #[D,N1,N2,C]->[D,N1,C,N2]->[D,C,N1,N2]

        super().__init__(data, labels, **perturb_kwargs)


    def set_perturb_mask(self, mask):
        """Sets self._perturb_mask to either None for random generation of the
        mask for each datapoint, or a fixed matrix same shape as one of the datapoints.
        """
        assert mask in ['first', 'last', 'rand']
        datapoint = self.data[0]
        if mask == 'first':
            self._perturb_mask = torch.zeros_like(datapoint, dtype=bool)
            #TODO: should we perturb the first K pix for *each* channel
            # or the first K/Cx pix for each channel
            # or the first K of the 1st channel or ... ?
            raise NotImplementedError()
        elif mask == 'last':
            self._perturb_mask = torch.zeros_like(datapoint, dtype=bool)
            raise NotImplementedError()
        elif mask == 'rand':
            self._perturb_mask = None #generated per datapoint
        else:
            raise ValueError('Invalid perturb mask')


    @staticmethod
    def batch_to_grid(batch):
        """batch is [B,C,W,H], where C=3 and W=H=32 for CIFAR10
        returns grid is [C, sqrt(B)*(W+1), sqrt(B)*(H+1)] (+1 is from padding)"""
        rows,_ = plots.length_to_rows_cols(len(batch))
        grid = torchvision.utils.make_grid(batch.detach(), rows, padding=1, normalize=True,
                                           pad_value=torch.tensor(float('nan')))
        grid = grid.transpose(0,1).transpose(1,2) #[C,W,H]->[W,C,H]->[W,H,C]
        return grid



def filter_by_class(data, labels, num_samples=None, select_classes='all', n_per_class='all', sort_by_class=True):
    if num_samples is not None:
        data = data[:num_samples]
        labels = labels[:num_samples]

    if select_classes == 'all':
        if n_per_class == 'all':
            return data, labels
        select_classes = labels.unique()

    selected_idx_per_class = []
    for c in select_classes:
        idx = (labels == c).nonzero().squeeze().view(-1) #get indices corresponding to this class
        if n_per_class != 'all':
            idx = idx[:n_per_class] #only take the first n_per_class items of this class
        selected_idx_per_class.append(idx)
    selected_idx = torch.cat((selected_idx_per_class))

    if num_samples is not None:
        assert len(selected_idx) == num_samples #sanity check

    if not sort_by_class:
        #revert sorting to original order in dataset
        selected_idx = selected_idx.sort()[0]

    return data[selected_idx], labels[selected_idx]



def get_data(**kwargs):
    DatasetClass = globals()[kwargs.pop('class')]
    include_test = kwargs.pop('include_test', False)

    train_data = DatasetClass(**kwargs)
    test_data = None

    if include_test:
        if DatasetClass == AssociativeRandom:
            test_data = DatasetClass(**kwargs)
        else:
            test_data = DatasetClass(test=True, **kwargs)

    return train_data, test_data



if __name__ == '__main__':
    data = AssociativeMNIST()
    data.plot_batch(num_samples=4)

    data = AssociativeCIFAR10(perturb_mask='rand')
    data.plot_batch(num_samples=4)
