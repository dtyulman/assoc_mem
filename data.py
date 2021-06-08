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



def get_aa_mnist_classification_data(include_test=False, balanced=True):
    data_transforms_list = [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1,1))]
    if balanced: #put data in range [+1,-1] instead of [0,1]
        data_transforms_list.append(transforms.Lambda(lambda x: 2*x-1 ))
    to_vec = transforms.Compose(data_transforms_list)
    to_onehot = transforms.Lambda(lambda y: torch.zeros(10,1)
                                  .scatter_(0, torch.tensor([[y]]), value=1))
    train_data = AutoassociativeDataset(
                    datasets.MNIST(root='./data/', download=True,
                                   transform=to_vec, target_transform=to_onehot)
                    )
    test_data = AutoassociativeDataset(
                    datasets.MNIST(root='./data/', train=False, download=True,
                                   transform=to_vec, target_transform=to_onehot)
                    )

    if test_data:
        return train_data, test_data
    return train_data
