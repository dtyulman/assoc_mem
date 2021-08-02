import itertools, warnings
from collections.abc import MutableMapping, Mapping
from copy import deepcopy


class Config(MutableMapping):
    """Nested dictionary that can be accessed by dot-separated keys
    e.g. cfg['a.b.c'] == cfg['a']['b']['c']"""
    #alternative: https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict/39375731#39375731
    #TODO: consider making immutable (replace set/delitem() with updated() method which returns a
    #copy with corresponding key added/removed/modified and change _storage to namedtuple)
    def __init__(self, input_dict):
        self._storage = {}
        for key, value in input_dict.items():
            if isinstance(value, Mapping):
                value = Config(value)
            self[key] = value

    def _get_storage_dict(self, keypath, create=False):
        if keypath == '':
            return self._storage
        storage = self._storage
        for key in keypath.split('.'):
            if create and key not in storage:
                storage[key] = Config({})
            storage = storage[key]._storage
        return storage

    def __getitem__(self, nestedkey):
        keypath, _, key = nestedkey.rpartition('.')
        return self._get_storage_dict(keypath)[key]

    def __setitem__(self, nestedkey, value):
        assert isinstance(nestedkey, str), 'Only string keys allowed'
        keypath, _, key = nestedkey.rpartition('.')
        self._get_storage_dict(keypath, create=True)[key] = value

    def __delitem__(self, nestedkey):
        keypath, _, key = nestedkey.rpartition('.')
        return self._get_storage_dict(keypath).__delitem__(key)

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


def flatten_config_loop(baseconfig, deltaconfigs, mode='combinatorial'):
    """With baseconfig as the default, return a list of configs with the entries in deltaconfigs.keys()
    looped over the corresponding lists in deltaconfigs.values()"""
    configs = []
    labels = []
    assert all([param in baseconfig for param in deltaconfigs]), \
        'Varied parameters must exist in baseconfig'
    if mode == 'combinatorial':
        for values in itertools.product(*deltaconfigs.values()):
            config = deepcopy(baseconfig)
            label = []
            for param, value in zip(deltaconfigs.keys(), values):
                config[param] = value
                label.append(f'{param.rpartition(".")[-1]}={value}')
            configs.append(config)
            labels.append(' '.join(label))
    elif mode == 'sequential':
        len_params_list = len(list(deltaconfigs.values())[0])
        assert all([len(values)==len_params_list for values in deltaconfigs.values()]), \
            'Parameter lists must be of same length'
        raise NotImplementedError()
    else:
        raise ValueError(f'Invalid mode: {mode}')
    return configs, labels


def verify_items(config, constraint, mode='raise'):
    """For each key in the constraint dict, asserts that config[key] is equal to that value. If not,
    raises an error or sets the config item to that value and issues a warning"""
    for key, value in constraint.items():
        try:
            satisfied_constraint = config[key] in value
        except:
            satisfied_constraint = config[key] == value

        if not satisfied_constraint:
            if mode == 'raise':
                raise ValueError(f"Config error: {key}={config[key]}, requires {value}")
            elif mode == 'set':
                warnings.warn(f"Config warning: setting {key}={config[key]} to {value}")
                config[key] = value
            else:
                raise ValueError(f"Invalid verify_item mode: '{mode}'")
    return config


def verify_config(config, mode='raise'):
    if config['data']['mode']['classify']:
        constraint = {
            'train.acc_fn' : 'cls',
            'train.acc_mode' : 'class',
            'data.mode.perturb_mode' : 'last',}
        if config['data.values.class'] == 'MNISTDataset':
            constraint.update({'data.mode.perturb_entries' : 10})
        elif config['data.values.class'] == 'RandomDataset':
            constraint.update({'data.mode.perturb_entries': config['data.values.num_classes']})
        else:
            raise ValueError(f"Invalid dataset class: {config['data.values.class']}")
    else:
        constraint = {
            # 'train.acc_fn' : 'mae',
            'train.acc_mode' : 'full',
            'train.loss_mode' : 'full',
            'data.values.normalize' : ['data', False]
            }

    return verify_items(config, constraint)


def load_config(path):
    with open(path, 'r') as f:
        config = eval(f.read())
    return config
