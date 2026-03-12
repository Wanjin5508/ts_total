from dataclasses import dataclass, field
import torch
from pathlib import Path
import numpy as np

TIME_INTERVAL = 20e-3
TIME_PER_SAMPLE = 16e-9
CHUNK_SIZE = int(np.ceil(TIME_INTERVAL / TIME_PER_SAMPLE))
SIGNAL_FREQ = 62.5e6    # S
TIME_SERIES_DATA_PATH = Path('')
DATA_INPUT_DIR = Path('')
DOWN_SAMPLING_KERNELS = [10, 8, 6, 4]

LOSS_WEIGHT = 0.8
SEG_FREQ = 62.5e3 # --> e
SAMPLE_POINTS_PER_SEG = int(SIGNAL_FREQ / SEG_FREQ) # i = S/e
SEG_NUM = int(CHUNK_SIZE / SAMPLE_POINTS_PER_SEG)    # T


@dataclass(frozen=False)
class ModelConfig:
    model_id:int
    num_classes:int = 2
    depth:int = 4
    random_seed:int = 42
    batch_size:int = 4
    epoch_num:int = 50
    learning_rate:float = 0.005
    
    loss_weight:float = 0.8
    segment_freq:float = 62.5e3
    device:str = 'cuda' if torch.cuda.is_available else 'cpu'
    check_points_path:Path = Path(__file__).resolve().parent.parent / 'outputs'
    
    sample_points_per_seg:int = field(init=False)
    seg_num:int = field(init=False)
    
def __post_init__(self):
    self.sample_points_per_seg = int(SIGNAL_FREQ / self.segment_freq)
    self.seg_num = int(CHUNK_SIZE / self.sample_points_per_seg)



