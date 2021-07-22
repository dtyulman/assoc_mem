from copy import deepcopy
import warnings

import numpy as np
import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset

############
# Datasets #
############
class ClassifyDatasetBase(Dataset):
    def __init__(self):
        self.input_size = None #TODO: enforce these get set at init in subclasses
        self.target_size = None
        self.num_classes = None

        self.data = None
        self.targets = None

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data)


class RandomDataset(ClassifyDatasetBase):
    def __init__(self, num_samples=50, input_size=20, num_classes=2, distribution='bern', **kwargs):
        self.input_size = input_size
        self.target_size = num_classes
        self.num_classes = num_classes

        _targets = torch.randint(num_classes,(num_samples,))
        self.targets = F.one_hot(_targets, num_classes=num_classes)

        if distribution == 'bern':
            p = kwargs.pop('p', 0.5)
            balanced = kwargs.pop('balanced', False)
            self.data = (torch.rand(num_samples, input_size, 1) > p).float()
            if balanced:
                self.data = 2*self.data - 1
        elif distribution == 'gaus':
            mean = kwargs.pop('mean', 0)
            std = kwargs.pop('std', 1)
            self.data = torch.randn(num_samples, input_size, 1)*std + mean
        elif distribution == 'unif':
            lo = kwargs.pop('lo', 0)
            hi = kwargs.pop('hi', 1)
            self.data = torch.rand(num_samples, input_size, 1)*(hi-lo) + lo
        else:
            raise ValueError(f"Invalid distribution spec: '{distribution}'")

        if kwargs:
            warnings.warn(f'Ignoring unused RandomDataset kwargs: {kwargs}')


class MNISTDataset(ClassifyDatasetBase):
    def __init__(self, num_samples=None, test=False, balanced=False, crop=False, downsample=False,
                 normalize=False, **kwargs):
        if kwargs:
            warnings.warn(f'Ignoring unused MNISTDataset kwargs: {kwargs}')

        #get data
        mnist = torchvision.datasets.MNIST(root='./data/', download=True, train=(not test))
        self.targets = mnist.targets
        self.data = mnist.data #in range [0, 255]

        #get subset
        if num_samples is not None:
            self.data = self.data[:num_samples]
            self.targets = self.targets[:num_samples]

        #apply target transformations
        self.num_classes = 10
        self.targets = F.one_hot(self.targets, num_classes=self.num_classes)
        self.target_size = self.targets.shape[1]

        #apply data transformations
        self.data = self.data/(self.data.max()-self.data.min()) #to range [0., 1.]
        if balanced: #put data in range [+1,-1] instead of [0,1]
            self.data = 2*self.data-1
        if crop: #remove `crop` pixels from top,bottom,left,right
            self.data = self.data[:, crop:-crop, crop:-crop]
        if downsample: #take every `downsample`th pixel vertically and horizontally
            self.data = self.data[:, ::downsample, ::downsample]
        num_samples, vpix, hpix = self.data.shape
        self.data = self.data.view(num_samples, vpix*hpix, 1) #flatten
        if normalize: #normalize each sample to unit vector
            self.data = self.data/self.data.norm(dim=1, keepdim=True)

        self.input_size = self.data.shape[1]


def get_data(**kwargs):
    DatasetClass = globals()[kwargs.pop('class')]
    include_test = kwargs.pop('include_test', False)

    train_data = DatasetClass(**kwargs)
    test_data = None

    if include_test:
        if DatasetClass == MNISTDataset:
            test_data = DatasetClass(test=True, **kwargs)
        elif DatasetClass == RandomDataset:
            test_data =  DatasetClass(**kwargs)

    return train_data, test_data


#######################
# Associative wrapper #
#######################
class AssociativeDataset(Dataset):
    def __init__(self, dataset, classify=False, perturb_mode='last', perturb_frac=None, perturb_num=None,
                 perturb_value=0, **kwargs):
        if kwargs:
            #kwargs ignores any extra values passed in (eg. via a Config object)
            warnings.warn(f'Ignoring unused AssociativeDataset kwargs: {kwargs}')

        self.dataset = dataset
        self.classify = classify
        self.input_size = dataset.input_size + dataset.target_size
        self.target_size = self.input_size

        assert (perturb_frac is None) != (perturb_num is None), \
            "Must specify either the fraction xor the number of entries to perturb"
        self.perturb_frac, self.perturb_num = perturb_frac, perturb_num
        if perturb_frac is None:
            self.perturb_frac = float(perturb_num) / self.input_size
        elif perturb_num is None:
            self.perturb_num = int(self.input_size*self.perturb_frac)

        assert perturb_mode in ['rand', 'first', 'last']
        self.perturb_mode = perturb_mode

        assert np.isreal(perturb_value) or perturb_value in ['rand', 'flip']
        self.perturb_value = perturb_value


    def _perturb(self, datapoint):
        datapoint = deepcopy(datapoint) #prevent in-place modification

        #perturb_mask indicates which entries will be perturbed
        if self.perturb_mode == 'rand':
            perturb_mask = torch.rand_like(datapoint) < self.perturb_frac
        elif self.perturb_mode in ['first', 'last']:
            #TODO: only need to compute perturb_mask once in this case, and can directly store
            #inputs as perturbed targets instead of perturbing on-the-fly
            perturb_mask = torch.zeros(datapoint.shape, dtype=bool)
            if self.perturb_mode == 'first':
                perturb_mask[:self.perturb_num] = 1
            elif self.perturb_mode == 'last':
                perturb_mask[-self.perturb_num:] = 1

        #perturb_value indicates what the perturbed entries will be set to
        if self.perturb_value == 'rand':
            datapoint[perturb_mask] = torch.rand_like(datapoint)[perturb_mask]
        elif self.perturb_value == 'flip':
            datapoint[perturb_mask] = -datapoint[perturb_mask]
        else:
            datapoint[perturb_mask] = self.perturb_value

        return datapoint, perturb_mask


    def __getitem__(self, idx):
        input, target = self.dataset[idx] #Md, Mc
        if self.classify:
            target = torch.cat((input, target)) #M=Md+Mc, else M=Md
        else:
            target = input
        input, perturb_mask = self._perturb(target) #entries input[perturb_mask] are perturbed
        return input, target, perturb_mask


    def __len__(self):
        return len(self.dataset)


def get_aa_data(data_kwargs, aa_kwargs):
    train_data, test_data = get_data(**data_kwargs)
    train_data = AssociativeDataset(train_data, **aa_kwargs)
    if test_data:
        test_data = AssociativeDataset(test_data, **aa_kwargs)
    return train_data, test_data


def get_aa_debug_batch(train_data, select_classes='all', n_per_class=None):
    debug_data = deepcopy(train_data) #train_data is an AssociativeDataset object
    debug_data.dataset = filter_classes(debug_data.dataset, select_classes=select_classes,
                                        n_per_class=n_per_class)
    return next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))


###########
# Helpers #
###########
def filter_classes(dataset, select_classes='all', n_per_class=None, sort_by_class=True):
    if select_classes == 'all':
        select_classes = torch.arange(dataset.num_classes)

    targets = dataset.targets.nonzero()[:,1] #get indices corresponding to this class

    selected_idx_per_class = []
    for c in select_classes:
        idx = (targets == c).nonzero().squeeze()
        if n_per_class is not None:
            idx = idx[:n_per_class] #only take the first n_per_class items of this class
        selected_idx_per_class.append(idx)
    selected_idx = torch.cat((selected_idx_per_class))

    if not sort_by_class:
        #revert sorting to original order in dataset
        selected_idx = selected_idx.sort()[0]

    filtered_dataset = deepcopy(dataset)
    filtered_dataset.data = dataset.data[selected_idx]
    filtered_dataset.targets = dataset.targets[selected_idx]
    return filtered_dataset
