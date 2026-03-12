import torch 
import numpy as np 

from torch.utils.data import Dataset 
from scipy import signal
from pathlib import Path


class TCNAEDataset(Dataset):
    """Create a windowed view of a signal stream for the training of TCNAE
    
    Args:

    """
    def __init__(self, 
                 path_to_signal: Path, 
                 n_samples:int=None, 
                 window_size:int=128, 
                 window_shift:int=128//2, 
                 filter=None):
        """Constructor of TCNAEDataset.
        
        Args:
            path_to_signal (Path): Path object leading to the desired file 
            n_samples (_type_, optional): Maximum number of sample points to read from the file. Defaults to slice (None)
            window_size (int, optional): Number of samples per window in Dataset
            window_shift (int, optional): Number of sample points to shift for next window in dataset
            filter (tuple, optional): Filtercoefficients b and a for a scipy.signal lfilter. 
        """
        
        super(Dataset, self).__init__()
        _file_ext = path_to_signal.suffix
        if _file_ext == '.npz':
            self.raw_data = np.load(path_to_signal)['adc_data']['ch0']
        elif _file_ext == '.npy':
            self.raw_data = np.load(path_to_signal).astype(np.float64)  # maybe 32
        elif _file_ext == '.txt':
            self.raw_data = np.genfromtxt(path_to_signal)
        else:
            raise Exception("You have to provide a path_to_signal with format .npy, .npz or .txt")
        
        if n_samples:
            self.raw_data = self.raw_data[:n_samples]
        if filter:
            self.raw_data = signal.lfilter(b=filter[0], a=filter[1], x=self.raw_data)
            
        
        # Preprocessing
        self.raw_data = scale(int_to_volt(self.raw_data))
        
        # Generate a list of non-overlapping windowed view
        _data = np.lib.stride_tricks.sliding_window_view(self.raw_data, window_size)[::window_shift]
        
        self.data = torch.from_numpy(np.copy(_data)).float()
        self.window_size = window_size
        
    def __len__(self):
        """_summary_
        
        Returns:
            int: number of windowed views in the dataset
        """
        return self.data.shape[0]

    def __getitem__(self, index:int):
        """_Summary_
        
        Args:
            index (int): index of the window 
            
        Returns:
            tensor: the windowed view of dataset with corresponding index
        """
        return self.data[index].reshape(1, self.window_size), self.data[index].reshape(1, self.window_size)


def int_to_volt(data, fgain=72):
    return (data / (2 ** 15)) * (40 / 10 ** (fgain/20))


def scale(arr: np.ndarray):
    return (arr - np.median(arr)) / (np.quantile(arr, 0.75) - np.quantile(arr, 0.25) + 1e-8)


