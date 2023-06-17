import torch
import numpy as np
import argparse

def set_torch_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(seconds):
    if seconds<1e-6:
        return "%dns" % (seconds*1e9)
    if seconds<1e-3:
        return "%dus" % (seconds*1e6)
    elif seconds<1:
        return "%dms" % (seconds*1e3)
    elif seconds<5:
        seconds_int = int(seconds)
        return "%ds%02dms" % (seconds, (seconds-seconds_int)*1000)
    else:
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "%dh%02dm%02ds" % (hours, minutes, seconds)

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')





