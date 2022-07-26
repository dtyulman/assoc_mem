import itertools, warnings, os, datetime, glob
from collections.abc import MutableMapping, Mapping
from copy import deepcopy

import networks, utils

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


def load_config(path):
    with open(path, 'r') as f:
        config = eval(f.read())
    return config


def initialize_savedir(baseconfig):
    root = os.path.dirname(os.path.abspath(__file__))
    ymd = datetime.date.today().strftime('%Y-%m-%d')
    saveroot = os.path.join(root, 'results', ymd)
    try:
        prev_run_dirs = glob.glob(os.path.join(saveroot, '[0-9]*'))
        prev_run_dirs = sorted([os.path.split(d)[-1] for d in prev_run_dirs])
        run_number = int(prev_run_dirs[-1])+1
    except (FileNotFoundError, IndexError, ValueError):
        run_number = 0
    savedir = os.path.join(saveroot, '{:04d}'.format(run_number))
    os.makedirs(savedir)
    with open(os.path.join(savedir, 'baseconfig.txt'), 'w') as f:
        f.write(repr(baseconfig)+'\n')
    print(f'Saving to: {savedir}')
    return savedir


def flatten_config_loop(baseconfig, deltaconfigs, mode='combinatorial'):
    """With baseconfig as the default, return a list of configs with the entries in deltaconfigs.keys()
    looped over the corresponding lists in deltaconfigs.values()"""

    assert all([param in baseconfig for param in deltaconfigs]), \
        'Varied parameters must exist in baseconfig'


    values_unzipped = [()]
    if len(deltaconfigs) > 0:
        if mode == 'combinatorial':
            values_unzipped = itertools.product(*deltaconfigs.values())
        elif mode == 'sequential':
            len_params_list = len(list(deltaconfigs.values())[0])
            assert all([len(values)==len_params_list for values in deltaconfigs.values()]), \
                'Parameter lists must be of same length'
            values_unzipped = zip(*deltaconfigs.values())
        else:
            raise ValueError(f'Invalid mode: {mode}')

    configs = []
    labels = []
    for values in values_unzipped:
        config = deepcopy(baseconfig)
        label = []
        for param, value in zip(deltaconfigs.keys(), values):
            config[param] = value
            label.append(f'{param.rpartition(".")[-1]}={value}')
        configs.append(config)
        labels.append(' '.join(label))

    return configs, labels


# def verify_items(config, constraint, mode='raise'):
#     """For each key in the constraint dict, asserts that config[key] is equal to that value. If not,
#     raises an error or sets the config item to that value and issues a warning"""
#     for key, value in constraint.items():
#         try:
#             satisfied_constraint = config[key] in value
#         except:
#             satisfied_constraint = config[key] == value

#         if not satisfied_constraint:
#             if mode == 'raise':
#                 raise ValueError(f"Config error: {key}={config[key]}, requires {value}")
#             elif mode == 'set':
#                 warnings.warn(f"Config warning: setting {key}={config[key]} to {value}")
#                 config[key] = value
#             else:
#                 raise ValueError(f"Invalid verify_item mode: '{mode}'")
#     return config


if __name__ == '__main__':
    def test_Config():
        #test init
        d = {'a': 'a0',
             'b' : {'1':'b1',
                    '2':'b2'},
             'b.3': 'b3',
             'c': {'1': {'x':'c1x',
                         'y':'c1y'},
                   '2.x': 'c2x',
                   '3.y': 'c3y'},
             'c.4': {'x':'c4x',
                     'y':'c4y'},
             'c.4.z' : 'c4z',
             'c.5.x' : 'c5x',
             }

        cfg = Config(d)
        assert cfg['a'] == 'a0'
        assert cfg['b.3'] == 'b3'
        assert cfg['c.3'] == Config({'y': 'c3y'})
        assert cfg['c.4.z'] == 'c4z'
        assert cfg['c.4']['z'] == 'c4z'

        #test delete
        assert cfg.pop('c.4.z') == 'c4z'
        assert cfg.pop('b') == Config({'1': 'b1', '2': 'b2', '3': 'b3'})

        #test insert
        cfg['d.1.x.ii'] = 'd1xii'
        assert cfg['d.1.x.ii'] == 'd1xii'

        #throws AssertionError
        try:
            cfg['x']
        except KeyError as e:
            assert e.args[0] == 'x'

        print('test_Config: Pass')


    def test_flatten_config_loop():
        baseconfig = Config({'a': 1, 'b': {'1':'p', '2':'q'}, 'c': None})
        deltaconfigs = {'a': [1,2,3],
                        'b.1': ['y','z']}

        #correct
        cfgs, labels = flatten_config_loop(baseconfig, deltaconfigs)
        expectedconfigs = [Config({'a': 1, 'b.1': 'y', 'b.2': 'q', 'c': None}),
                           Config({'a': 1, 'b.1': 'z', 'b.2': 'q', 'c': None}),
                           Config({'a': 2, 'b.1': 'y', 'b.2': 'q', 'c': None}),
                           Config({'a': 2, 'b.1': 'z', 'b.2': 'q', 'c': None}),
                           Config({'a': 3, 'b.1': 'y', 'b.2': 'q', 'c': None}),
                           Config({'a': 3, 'b.1': 'z', 'b.2': 'q', 'c': None})]
        expectedlabels = ['a=1_1=y',
                          'a=1_1=z',
                          'a=2_1=y',
                          'a=2_1=z',
                          'a=3_1=y',
                          'a=3_1=z']
        assert cfgs == expectedconfigs, f'Output:{cfgs}, Expected:{expectedconfigs}'
        assert labels == expectedlabels, f'Output:{labels}, Expected:{expectedlabels}'

        #throws AssertionError
        try:
            deltaconfigs_bad = {'d': '1'} #fails because not in base
            cfgs, labels = flatten_config_loop(baseconfig, deltaconfigs_bad)
        except AssertionError as e:
            assert e.args[0] == 'Varied parameters must exist in baseconfig'
        print('test_flatten_config_loop: Pass')


    test_Config()
    test_flatten_config_loop()
