from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class AssociativeDataset(Dataset):
    def __init__(self, dataset, perturb_frac=0.5, perturb_mode='rand', perturb_value=0.):
        self.dataset = dataset
        self.input_size = self.target_size = dataset[0][0].numel()

        self.perturb_frac = perturb_frac
        assert perturb_mode in ['rand', 'first', 'last']
        self.perturb_mode = perturb_mode
        if isinstance(perturb_value, str):
            assert perturb_value in ['rand', 'flip']
        self.perturb_value = perturb_value


    def _perturb(self, datapoint):
        #perturb_mask indicates which entries will be perturbed
        datapoint = deepcopy(datapoint)

        if self.perturb_mode == 'rand':
            perturb_mask = torch.rand_like(datapoint) < self.perturb_frac
        elif self.perturb_mode in ['first', 'last']:
            perturb_len = int(len(datapoint)*self.perturb_frac)
            perturb_mask = torch.zeros(datapoint.shape, dtype=bool)
            if self.perturb_mode == 'first':
                perturb_mask[:perturb_len] = 1
            elif self.perturb_mode == 'last':
                perturb_mask[-perturb_len:] = 1

        if self.perturb_value == 'rand':
            datapoint[perturb_mask] = torch.rand_like(datapoint)[perturb_mask]
        elif self.perturb_value == 'flip':
            datapoint[perturb_mask] = -datapoint[perturb_mask]
        else:
            datapoint[perturb_mask] = self.perturb_value

        return datapoint


    def __getitem__(self, idx):
        target, _ = self.dataset[idx]
        input = self._perturb(target)
        return input, target


    def __len__(self):
        return len(self.dataset)



class AssociativeClassifyDataset(Dataset):
    """
    Convert a dataset with (input, target) pairs to work with autoassociative
    memory networks, returning (aa_input, aa_target) pairs, where
    aa_input = [input, init] and aa_target = [input, target]
    """
    def __init__(self, dataset, output_init_value=0):
        self.dataset = dataset
        self.output_init_value = output_init_value
        self.input_size = dataset[0][0].numel() # Md
        self.target_size = dataset[0][1].numel() # Mc

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        output_init = torch.full_like(target, self.output_init_value)
        aa_input = torch.cat((input, output_init)) # M=Md+Mc
        aa_target = torch.cat((input, target)) # M=Md+Mc
        return aa_input, aa_target

    def __len__(self):
        return len(self.dataset)



def get_aa_mnist_data(include_test=False, balanced=False, crop=False, downsample=False,
                       mode='classify', **kwargs):
    if mode == 'classify':
        DatasetClass = AssociativeClassifyDataset
    elif mode == 'complete':
        DatasetClass = AssociativeDataset

    train_data, test_data = get_mnist_data(include_test, balanced, crop, downsample)
    train_data = DatasetClass(train_data, **kwargs)
    if test_data:
        test_data = DatasetClass(test_data, **kwargs)
    return train_data, test_data


def get_aa_debug_batch(train_data, select_classes='all', n_per_class=None):
    #train_data is an AssociativeDataset or AssociativeClassifyDataset object
    debug_data = deepcopy(train_data)
    debug_data.dataset = filter_classes(debug_data.dataset, select_classes=select_classes,
                                        n_per_class=n_per_class)

    debug_input, debug_target = next(iter(torch.utils.data.DataLoader(debug_data,
                                                                      batch_size=len(debug_data))
                                          ))
    return debug_input, debug_target


def get_mnist_data(include_test=False, balanced=False, crop=False, downsample=False):
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
