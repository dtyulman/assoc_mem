from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class AutoassociativeDataset(Dataset):
    """
    Convert a dataset with (input, target) pairs to work with autoassociative
    memory networks, returning (aa_input, aa_target) pairs, where
    aa_input = [input, init] and aa_target = [input, target]
    """
    def __init__(self, dataset, output_init_value=0):
        self.dataset = dataset
        self.output_init_value = output_init_value
        self.input_size = dataset[0][0].numel() # Mv
        self.target_size = dataset[0][1].numel() # Mc

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        output_init = torch.full_like(target, self.output_init_value)
        aa_input = torch.cat((input, output_init)) # M=Mv+Mc
        aa_target = torch.cat((input, target)) # M=Mv+Mc
        return aa_input, aa_target

    def __len__(self):
        return len(self.dataset)


def get_aa_mnist_classification_data(include_test=False, balanced=False, crop=False, downsample=False):
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
    train_data = AutoassociativeDataset(train_data)
    if include_test:
        test_data = datasets.MNIST(root='./data/', train=False, download=True, transform=to_vec,
                                   target_transform=to_onehot)
        test_data = AutoassociativeDataset(test_data)
        return train_data, test_data
    return train_data


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
