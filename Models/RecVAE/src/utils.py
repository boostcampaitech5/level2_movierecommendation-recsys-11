import os
import random
import numpy as np
import torch


def set_seeds(seed: int = 4948):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### RecVAE ###

def generate(args, batch_size, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
  
    N = data_in.shape[0]
    samples_per_epoch = int(N * samples_perc_per_epoch)
  
    if shuffle:
        idx_list = np.arange(N)
        np.random.shuffle(idx_list)
        idx_list = idx_list[:samples_per_epoch]
    else:
        idx_list = np.arange(samples_per_epoch)
  
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idx_list[st_idx:end_idx]
    
        yield Batch(args, idx, data_in, data_out)
    
class Batch:
    def __init__(self, args, idx, data_in, data_out=None):
        self._device = args.device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
  
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
  
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
  
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
        self.get_ratings(is_out).toarray()
        ).to(self._device)
        

def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}