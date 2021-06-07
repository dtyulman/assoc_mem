import time

class Timer():
    def __init__(self, name=None):
        self.name = name

    def __enter__(self, name=None):
        print('Starting timer{}'.format(f': {name}...' if name is not None
                                        else '...'))
        self.start_time = time.perf_counter()

    def __exit__(self, *args):
        stop_time = time.perf_counter()
        elapsed = stop_time - self.start_time
        elapsed_str = f'Time elapsed: {elapsed}'
        if self.name is not None:
            elapsed_str = f'{self.name}: {elapsed_str} sec'
        print(elapsed_str)
