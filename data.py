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

    def _normalize_init(self, mode):
        if mode == 'data': #normalize each sample to unit vector
            self.data = self.data/self.data.norm(dim=1, keepdim=True)
        elif mode == 'data+targets':
            norm = torch.cat((self.data, self.targets), dim=1).norm(dim=1, keepdim=True)
            self.data = self.data/norm
            self.targets = self.targets/norm
        elif mode != False:
            raise ValueError(f'Invalid normalization mode: {mode}')



class RandomDataset(ClassifyDatasetBase):
    def __init__(self, num_samples=50, input_size=20, num_classes=2, normalize=False,
                 distribution='bern', **kwargs):
        self.input_size = input_size
        self.target_size = num_classes
        self.num_classes = num_classes

        _targets = torch.randint(num_classes,(num_samples,))
        self.targets = F.one_hot(_targets, num_classes=num_classes).unsqueeze(-1)

        if distribution == 'bern':
            p = kwargs.pop('p', 0.5)
            balanced = kwargs.pop('balanced', False)
            self.data = (torch.rand(num_samples, input_size, 1) > p).to(torch.get_default_dtype())
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

        self._normalize_init(normalize)



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
        self.targets = F.one_hot(self.targets, num_classes=self.num_classes).unsqueeze(-1)
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
        self.data = self.data.reshape(num_samples, vpix*hpix, 1) #flatten
        self._normalize_init(normalize) #maybe normalize

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
            test_data = DatasetClass(**kwargs)

    return train_data, test_data


#######################
# Associative wrapper #
#######################
class AssociativeDataset(Dataset):
    def __init__(self, dataset, classify=False, perturb_mode='last', perturb_entries=0.5,
                 perturb_value=0, **kwargs):
        if kwargs:
            #kwargs ignores any extra values passed in (eg. via a Config object)
            warnings.warn(f'Ignoring unused AssociativeDataset kwargs: {kwargs}')

        self.dataset = dataset
        self.classify = classify

        self.input_size = dataset.input_size
        if self.classify:
            self.input_size += dataset.target_size
        self.target_size = self.input_size

        if perturb_entries < 1: #specified the fraction of entries to perturb
            self.perturb_frac = perturb_entries
            self.perturb_num = int(self.input_size*self.perturb_frac)
        else: #specified the number of entries to perturb
            self.perturb_num = perturb_entries
            self.perturb_frac = float(self.perturb_num) / self.input_size
        assert 0 < self.perturb_frac < 1, \
            'Must have 0 < perturb_frac ({self.perturb_frac}) < 1'
        assert self.perturb_num < self.input_size, \
            'Must have perturb_num ({self.perturb_num}) < input_size ({self.input_size})'


        assert perturb_mode in ['rand', 'first', 'last']
        self.perturb_mode = perturb_mode

        assert np.isreal(perturb_value) or perturb_value in ['rand', 'flip', 'min', 'max']
        if perturb_value == 'min':
            self.perturb_value = self.dataset.data.min()
        elif perturb_value == 'max':
            self.perturb_value == self.dataset.data.max()
        else:
            self.perturb_value = perturb_value


    def _perturb(self, datapoint):
        datapoint = deepcopy(datapoint) #prevent in-place modification

        #perturb_mask indicates which entries will be perturbed
        if self.perturb_mode == 'rand':
            if len(datapoint.shape)==3: #[B,M,1]
                perturb_mask = torch.rand_like(datapoint[0]) < self.perturb_frac
                perturb_mask = perturb_mask.tile(datapoint.shape[0], 1, 1)
            elif len(datapoint.shape)==2: #[M,1]
                perturb_mask = torch.rand_like(datapoint) < self.perturb_frac

        elif self.perturb_mode in ['first', 'last']:
            #TODO: only need to compute perturb_mask once in this case, and can directly store
            #inputs as perturbed targets instead of perturbing on-the-fly
            perturb_mask = torch.zeros_like(datapoint, dtype=bool)
            if self.perturb_mode == 'first':
                if len(datapoint.shape)==3: #[B,M,1]
                    perturb_mask[:,:self.perturb_num] = 1
                elif len(datapoint.shape)==2: #[M,1]
                    perturb_mask[:self.perturb_num] = 1
            elif self.perturb_mode == 'last':
                if len(datapoint.shape)==3: #[B,M,1]
                    perturb_mask[:,-self.perturb_num:] = 1
                elif len(datapoint.shape)==2: #[M,1]
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
        input, target = self.dataset[idx] #[len(idx),Md,1], [len(idx),Mc,1] or [Md,1], [Mc,1]
        if self.classify:
            target = torch.cat((input, target), dim=-2) #M=Md+Mc, else M=Md
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

    targets = dataset.targets.nonzero()[:,1] #one-hot to index

    selected_idx_per_class = []
    for c in select_classes:
        idx = (targets == c).nonzero().squeeze().view(-1) #get indices corresponding to this class
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


class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = deepcopy(dataset)
        del dataset
        torch.cuda.empty_cache()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()


    def reset(self):
        if self.shuffle:
            perm = torch.randperm(len(self.dataset))
            self.dataset.dataset.data = self.dataset.dataset.data[perm]
            self.dataset.dataset.targets = self.dataset.dataset.targets[perm]
        self.i = 0
        self.j = min(self.i + self.batch_size, len(self.dataset))


    def __next__(self):
        if self.i < len(self.dataset):
            batch = self.dataset[self.i : self.j]
            self.i = self.j
            self.j = min(self.i + self.batch_size, len(self.dataset))
            return batch
        else:
            self.reset()
            raise StopIteration()

    def __iter__(self):
        return self
