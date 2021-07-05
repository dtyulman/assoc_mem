import configs
#TODO: unit test

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

    cfg = configs.Config(d)
    assert cfg['a'] == 'a0'
    assert cfg['b.3'] == 'b3'
    assert cfg['c.3'] == configs.Config({'y': 'c3y'})
    assert cfg['c.4.z'] == 'c4z'
    assert cfg['c.4']['z'] == 'c4z'

    #test delete
    assert cfg.pop('c.4.z') == 'c4z'
    assert cfg.pop('b') == configs.Config({'1': 'b1', '2': 'b2', '3': 'b3'})

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
    baseconfig = configs.Config({'a': 1, 'b': {'1':'p', '2':'q'}, 'c': None})
    deltaconfigs = {'a': [1,2,3],
                    'b.1': ['y','z']}

    #correct
    cfgs, labels = configs.flatten_config_loop(baseconfig, deltaconfigs)
    expectedconfigs = [configs.Config({'a': 1, 'b.1': 'y', 'b.2': 'q', 'c': None}),
                       configs.Config({'a': 1, 'b.1': 'z', 'b.2': 'q', 'c': None}),
                       configs.Config({'a': 2, 'b.1': 'y', 'b.2': 'q', 'c': None}),
                       configs.Config({'a': 2, 'b.1': 'z', 'b.2': 'q', 'c': None}),
                       configs.Config({'a': 3, 'b.1': 'y', 'b.2': 'q', 'c': None}),
                       configs.Config({'a': 3, 'b.1': 'z', 'b.2': 'q', 'c': None})]
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
        cfgs, labels = configs.flatten_config_loop(baseconfig, deltaconfigs_bad)
    except AssertionError as e:
        assert e.args[0] == 'Varied parameters must exist in baseconfig'

    print('test_flatten_config_loop: Pass')



if __name__ == '__main__':
    test_Config()
    test_flatten_config_loop()
