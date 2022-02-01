from pathlib import Path
from yacs.config import CfgNode as CN
import os
import time
import logging
import torch.distributed as dist



_C = CN()
_C.dataset = 'imagenet'
_C.data_dir = './data_list/'
_C.check_path = './checkpoint'
_C.arch = 'resnet50'
_C.workers = 32
_C.epochs = 400
_C.defer_epoch = 0
_C.start_epoch = 1
_C.batch_size = 256
_C.lr = 0.02
_C.momentum = 0.9
_C.weight_decay = 5e-4
_C.print_freq = 100
_C.resume = ''
_C.resume2 = ''
_C.world_size = 1
_C.rank = 0
_C.dist_url = 'tcp://localhost:10000'
_C.dist_backend = 'nccl'
_C.seed = None
_C.gpu = None
_C.evaluate = False
_C.multiprocessing_distributed = True


# options for moco v2
_C.moco_dim = 128
_C.moco_k = 8192
_C.moco_m = 0.999
_C.grad = False
_C.mlp = True
_C.aug_plus = False
_C.normalize = False
_C.queue_size_per_cls = 4
_C.smooth = 0.1
_C.ldam_m = 0.1

# options for SupCon
_C.con_type = 'SupConLoss'
_C.gamma = 128
_C.margin = 0.25

_C.con_weight = 1.0
_C.balsfx_n = 0.0
_C.effective_num_beta = 0.99
_C.temperature = 0.1

_C.log_weight = 7.0

# options for others
_C.mark = ''
_C.debug = False
_C.aug = 'randcls_sim'
_C.log_dir = 'logs'
_C.model_dir = 'ckps'
_C.warm_epochs = 10

_C.randaug_m = 10
_C.randaug_n = 2
_C.color_p = 1.0
_C.color_h = 0.0

_C.branch_type = 'balance'
_C.alpha = 0.2
_C.path = 'same'

_C.pos_size_per_cls = 8 
_C.neg_size_per_cls = 4


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    log_dir = Path("saved")  / (cfg.mark) / Path(cfg.log_dir)
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.txt'.format(cfg.mark)
    # final_log_file = log_dir / log_file 
    
    model_dir =  Path("saved") / (cfg.mark) / Path(cfg.model_dir)
    print('=> creating {}'.format(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir = str(model_dir)

    # cfg.freeze()

import logging
import os
import sys


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op


def get_logger(config, resume=False, is_rank0=True):
    """Get the program logger.
    Args:
        log_dir (str): The directory to save the log file.
        log_name (str, optional): The log filename. If None, it will use the main
            filename with ``.log`` extension. Default is None.
        resume (str): If False, open the log file in writing and reading mode.
            Else, open the log file in appending and reading mode; Default is "".
        is_rank0 (boolean): If True, create the normal logger; If False, create the null
           logger, which is useful in DDP training. Default is True.
    """
    if is_rank0:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        # # StreamHandler
        # stream_handler = logging.StreamHandler(sys.stdout)
        # stream_handler.setLevel(level=logging.INFO)
        # logger.addHandler(stream_handler)

        # FileHandler
        if resume == False:
            mode = "w+" 
        else:
            mode = "a+"
        log_dir = Path("saved") / (config.mark) / Path(config.log_dir)
        log_name = config.mark + ".log"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode=mode)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
    else:
        logger = NoOp()

    return logger