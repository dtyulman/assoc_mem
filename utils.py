import os, glob, datetime, gc, warnings
import torch, joblib


def initialize_savedir(baseconfig):
    # TODO: folder structure should be
    # assoc_mem/results/
    # | -- yyyy-mm-dd/
    # | | -- nnnn/
    # | | | -- baseconfig.txt
    # | | | -- deltaconfiglabel/
    # | | | | -- net.pt
    # | | | | -- log.pkl
    # | | | | -- config.pkl
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


def choose_device(dev_str):
    """If dev_str is 'cuda', returns 'cuda:X' where X is the gpu with the most free memory.
    """
    assert dev_str.startswith('cuda') or dev_str=='cpu', "Device must be 'cuda:[number]' or 'cpu'"
    if dev_str == 'cuda':
        if not torch.cuda.is_available():
            warnings.warn('CUDA not available. Using CPU.')
            return 'cpu'

        try:
            import pynvml
        except ImportError as e:
            warnings.warn(e.msg + f'. Using default device: {dev_str}.')
            return dev_str
        best_dev_idx = 0
        best_mem_free = 0
        pynvml.nvmlInit()
        for i in range(torch.cuda.device_count()):
            device = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_free = pynvml.nvmlDeviceGetMemoryInfo(device).free
            if mem_free > best_mem_free:
                best_mem_free = mem_free
                best_dev_idx = i
        dev_str = f'cuda:{best_dev_idx}'
    return dev_str


def load_from_dir(savedir):
    net = torch.load(os.path.join(savedir, 'net.pt'))
    logger = joblib.load(os.path.join(savedir, 'log.pkl'))
    config = joblib.load(os.path.join(savedir, 'config.pkl'))

    return net, logger, config


def mem_report():
    #https://gist.github.com/Stonesjtu/368ddf5d9eb56669269ecdf9b0d21cbe
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)
