from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class AssociativeDataset(Dataset):
    def __init__(self, dataset, mode='complete', perturb_mode='last',
                 perturb_frac=None, perturb_num=None,  perturb_value=0):
        self.dataset = dataset
        assert mode in ['classify', 'complete'], f"Invalid data mode: '{mode}'"
        self.mode = mode

        self.input_size = dataset[0][0].numel()
        if mode == 'complete':
            self.target_size = self.input_size
        elif mode == 'classify':
            self.target_size = dataset[0][1].numel()

        assert (perturb_frac==None) != (perturb_num==None), \
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
        elif self.perturb_mode in ['first', 'last']: #TODO: only need to compute perturb_mask once
            perturb_mask = torch.zeros(datapoint.shape, dtype=bool)
            if self.perturb_mode == 'first':
                perturb_mask[:self.perturb_num] = 1
            elif self.perturb_mode == 'last':
                perturb_mask[-self.perturb_num:] = 1

        #perturb_value indicates what the perturbed entried will be set to
        if self.perturb_value == 'rand':
            datapoint[perturb_mask] = torch.rand_like(datapoint)[perturb_mask]
        elif self.perturb_value == 'flip':
            datapoint[perturb_mask] = -datapoint[perturb_mask]
        else:
            datapoint[perturb_mask] = self.perturb_value

        return datapoint, perturb_mask


    def __getitem__(self, idx):
        input, target = self.dataset[idx] #Md, Mc
        if self.mode == 'classify':
            input = torch.cat((input, target)) #M=Md+Mc, else M=Md

        target = input #target is unperturbed version
        input, perturb_mask = self._perturb(input) #entries input[perturb_mask] are perturbed
        return input, target, perturb_mask


    def __len__(self):
        return len(self.dataset)



def get_aa_mnist_data(mnist_kwargs, aa_kwargs):
    train_data, test_data = get_mnist_data(**mnist_kwargs)

    train_data = AssociativeDataset(train_data, **aa_kwargs)
    if test_data:
        test_data = AssociativeDataset(test_data, **aa_kwargs)
    return train_data, test_data



def get_aa_debug_batch(train_data, select_classes='all', n_per_class=None):
    #train_data is an AssociativeDataset object
    debug_data = deepcopy(train_data)
    debug_data.dataset = filter_classes(debug_data.dataset, select_classes=select_classes,
                                        n_per_class=n_per_class)

    debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data, batch_size=len(debug_data))))
    return debug_input, debug_target



def get_mnist_data(include_test=False, balanced=False, crop=False, downsample=False, normalize=False):
    data_transforms_list = [transforms.ToTensor()]

    if balanced: #put data in range [+1,-1] instead of [0,1]
        data_transforms_list.append(transforms.Lambda(lambda x: 2*x-1 ))

    if crop: #remove <crop> pixels from top,bottom,left,right
        #note output of ToTensor is CxHxW so keep first dimension
        data_transforms_list.append(transforms.Lambda(lambda x: x[:,crop:-crop,crop:-crop]) )

    if downsample:
        data_transforms_list.append(transforms.Lambda(lambda x: x[:,::downsample,::downsample] ))

    #would like to do .view(-1,1) to avoid copy but throws error if cropping
    data_transforms_list.append( transforms.Lambda(lambda x: x.reshape(-1,1)) )

    if normalize:
        data_transforms_list.append(transforms.Lambda(lambda x: x/x.norm()))

    to_vec = transforms.Compose(data_transforms_list)
    to_onehot = transforms.Lambda(lambda y: torch.zeros(10,1)
                                  .scatter_(0, torch.tensor([[y]]), value=1))

    train_data = datasets.MNIST(root='./data/', download=True, transform=to_vec,
                                target_transform=to_onehot)
    test_data = datasets.MNIST(root='./data/', train=False, download=True, transform=to_vec,
                               target_transform=to_onehot) if include_test else None

    return train_data, test_data



def filter_classes(dataset, select_classes='all', n_per_class=None, sort_by_class=True):
    if select_classes == 'all':
        select_classes = dataset.class_to_idx.values()

    selected_idx_per_class = []
    for c in select_classes:
        idx = (dataset.targets == c).nonzero().squeeze() #get indices corresponding to this class
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
