import itertools
from collections.abc import MutableMapping, Mapping
from copy import deepcopy


class Config(MutableMapping):
    def __init__(self, input_dict):
        self._storage = {}
        for key, value in input_dict.items():
            if isinstance(value, Mapping):
                value = Config(value)
            self[key] = value

    def __getitem__(self, nestedkey):
        keypath, _, key = nestedkey.rpartition('.')
        return self._follow_key_path(keypath)[key]

    def __setitem__(self, nestedkey, value):
        assert isinstance(nestedkey, str), 'Only string keys allowed'
        keypath, _, key = nestedkey.rpartition('.')
        self._follow_key_path(keypath, create=True)[key] = value

    def __delitem__(self, nestedkey):
        keypath, _, key = nestedkey.rpartition('.')
        return self._follow_key_path(keypath).__delitem__(key)

    def __iter__(self):
        return self._storage.__iter__()

    def __len__(self):
        return self._storage.__len__()

    def __repr__(self, indent=8):
        name = type(self).__name__
        string = name + '({'
        for i, (key,value) in enumerate(self._storage.items()):
            key_repr = repr(key)
            if isinstance(value, Config):
                value_repr = value.__repr__(indent+10+len(key_repr))
            else:
                value_repr = repr(value)
            indent_str = indent*' ' if i>0 else ''
            string += f'{indent_str}{key_repr}: {value_repr},\n'
        string += indent_str + '})'
        return string

    def _follow_key_path(self, keypath, create=False):
        if keypath == '':
            return self._storage
        storage = self._storage
        for key in keypath.split('.'):
            if create and key not in storage:
                storage[key] = Config({})
            storage = storage[key]._storage
        return storage


def flatten_config_loop(baseconfig, deltaconfigs, mode='combinatorial'):
    """With baseconfig as the default, return a list of configs with the entries in deltaconfigs.keys()
    looped over the corresponding lists in deltaconfigs.values()"""
    configs = []
    labels = []
    assert all([param in baseconfig for param in deltaconfigs]), 'Varied parameters must exist in baseconfig'
    if mode == 'combinatorial':
        for values in itertools.product(*deltaconfigs.values()):
            config = deepcopy(baseconfig)
            label = []
            for param, value in zip(deltaconfigs.keys(), values):
                config[param] = value
                label.append(f'{param.rpartition(".")[-1]}={value}')
            configs.append(config)
            labels.append('_'.join(label))
    elif mode == 'sequential':
        len_params_list = len(list(deltaconfigs.values())[0])
        assert all([len(values)==len_params_list for values in deltaconfigs.values()]), 'Parameter lists must be of same length'
        raise NotImplementedError()
    else:
        raise ValueError(f'Invalid mode: {mode}')
    return configs, labels


###################
# Default configs #
###################
# Training
train_config = Config({
    'name': 'FPT', #FPT, SGD
    'batch_size': 50,
    'lr': 0.01, #for FPT only
    'print_every': 10,
    'loss_mode': 'full', # full, class
    'epochs':10000
    })


# Networks
net_config = Config({
    'class': 'ModernHopfield',
    'input_size': None, #if None, infer from dataset
    'hidden_size': 50,
    'beta': 1,
    'tau': 1,
    'input_mode': 'init', #init, cont, init+cont, clamp
    'dt': 0.05,
    'num_steps': 1000,
    'fp_mode': 'iter', #iter, del2
    'fp_thres':1e-9,
    })


# Datasets
data_config = Config({
    'name': 'MNIST',
    'subset': False, #positive integer takes only first N items
    'test': True
    })


###################
# Premade configs #
###################
def get_config(name='default'):
    if name == 'default':
        config = Config({
            'net': net_config,
            'data': data_config,
            'train': train_config
            })
    elif name == 'sgd':
        config = get_config()
        config['train']['name'] = 'SGD'
        del config['train']['lr']
    else:
        raise ValueError(f"Invalid config name: '{name}'")
    return deepcopy(config)
