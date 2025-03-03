import os, torch, random
import numpy as np
from datetime import datetime

class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.seed=100
        self.Learning_rate = 5e-5 
        self.Epoch = 200
        self.weight_decay = 0 
        self.embed_dim = 64
        self.protein_kernel = [7]
        self.drug_kernel = [4, 8, 12]
        self.conv = 160 #40
        self.protein_char_dim = 320
        self.drug_char_dim = 156  # molclr的embedding长度
        self.num_workers = 0
        self.num_features_xd = 78
        self.batch_size = 32                            # batch size
        self.num_workers = 0               
        self.target = 'Y'


class mcan_cfg():
    def __init__(self):
        super(mcan_cfg, self).__init__()
        self.LAYER = 1
        self.HIDDEN_SIZE = 160 #160
        self.MULTI_HEAD = 4
        self.DROPOUT_R = 0
        self.FF_SIZE = int(self.HIDDEN_SIZE * 2)
        self.HIDDEN_SIZE_HEAD = int(self.HIDDEN_SIZE / self.MULTI_HEAD)