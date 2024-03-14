import pickle
import gzip
from torch.utils.tensorboard import SummaryWriter
import copy
import os


def get_logger(cfg):
    if cfg.LOGGING.ENABLE_TENSORBOARD:
        return TensorboardLogger(cfg)
    else:
        return Logger(cfg)


class Logger:
    def __init__(self, cfg):
        self.data = {}
        self.cfg = cfg
        self.output_path = cfg.OUTPUT_PATH
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self):
        path = os.path.join(self.output_path, 'log')
        pickle.dump(self.data, gzip.open(path, 'wb'))


class TensorboardLogger(Logger):
    def __init__(self, cfg):
        self.data = {}
        self.context = ""
        self.cfg = cfg
        self.output_path = cfg.OUTPUT_PATH
        tb_dir = cfg.LOGGING.TB_DIR
        if tb_dir == '':
            tb_dir = os.path.join(cfg.OUTPUT_PATH, 'tb_log')

        self.writer = SummaryWriter(log_dir=tb_dir, comment=f"{cfg.LOGGING.TB_NAME}")

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]
        self.writer.add_scalar(key, value, len(self.data[key]))

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self):
        path = os.path.join(self.output_path, 'logs')
        pickle.dump(self.data, gzip.open(path, 'wb'))

    def draw_histogram(self, key, value, step):
        self.writer.add_histogram(key, value, step, bins='auto')
