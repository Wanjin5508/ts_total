import sys
import asyncio
import time
from pathlib import Path

import numpy as np
import polars as pl
import polars.datatypes as DataType
from typing import Tuple, Union, Literal, Optional

from export import OutputDF


class Dataset:
    """
    This class is used to load raw dataset
    """
    rate = 0
    chunk_cnt = 0
    chunk_idx = 0
    file_cnt = 0
    chunk_size = 0
    # file_path = ['data\meas_recid0_ich0_traidd_och0_ch0.npy', 'data\meas_recid1_ich0_traidd_och0_ch0.npy']
    file_path:list = []
    # array_0 = np.load('data\meas_recid0_ich0_traidd_och0_ch0.npy')
    # array_1 = np.load('data\meas_recid1_ich0_traidd_och0_ch0.npy')


    def __init__(self, chunk_size=1250000, sample_rate=1):  # chunk_size=1200000, sample_rate=4
        # self.path_0 = data_path_0
        # self.path_1 = data_path_1

        self.array_list = []
        if Dataset.file_path[0].split('.')[-1] == 'npy':

            # ! because of the header in .npy files, we have to set the offset to 128 >> dtypd is int16, so we need 64 * 2 = 120 Bytes
            # self.array_0 = np.memmap('C:/Users/liuw/code/data/meas_recid0_ich0_traidd_och0_ch0.npy', dtype='int16', mode='r', offset=128)
            # self.array_1 = np.memmap('C:/Users/liuw/code/data/meas_recid1_ich0_traidd_och0_ch0.npy', dtype='int16', mode='r', offset=128)
            for path in Dataset.file_path:
                array = np.load(path, mmap_mode='r')
                # array = np.memmap(path, dtype='int16', mode='r', offset=128)
                print(array.filename)
                self.array_list.append(array)

        else:
            # npz
            for path in Dataset.file_path:
                self.npz = Npz(path)
                self.array_list.append(self.npz.get_ch(0))


        Dataset.chunk_size = chunk_size
        Dataset.sample_rate = sample_rate
        Dataset.rate = sample_rate

        # if len(self.array_list) > 1:
        #     self.length = max(self.array_list[0].size, self.array_list[1].size)
        self.length = max([self.array_list[i].size for i in range(len(self.array_list))])
        self.chunk_cnt = int(self.length / self.chunk_size)

        self.sample = self.create_sample_data()
        self.output_df = OutputDF(len(self.array_list))


    def create_sample_data(self, chunk_idx:int=0) -> Tuple:

        Dataset.chunk_cnt = int(self.length / self.chunk_size)

        df_list = []
        # for chunk in chunk_per_source:
        #     chunk_per_source.append(array[start: end])

        results = asyncio.run( self.get_chunk_data(chunk_idx))
        # for file_path, chunk_id, data_chunk in results:
        for data_chunk in results:
            df_list.append(self.array2df(data_chunk, self.sample_rate))


        df_tuple = tuple(df_list)
        # print(f"async Dataset.chunk_cnt = {Dataset.chunk_cnt}")
        # print(f"create sample data shape: {df_list[0].shape}")

        return df_tuple


    async def read_chunk(self, array, chunk_id):
        start = chunk_id * self.chunk_size
        end = start + self.chunk_size

        start_time = time.time()
        data_chunk = array[start: end]
        await asyncio.sleep(0)
        end_time = time.time()
        # print(f"Chunk {chunk_id} from {array.filename} read at [start_time: .4f}s, completed at {end_time: .4f}s")
        # return array.filename, chunk_id, data_chunk
        return data_chunk


    async def get_chunk_data(self, chunk_id):
        tasks = [self.read_chunk(array, chunk_id) for array in self.array_list]
        results = await asyncio.gather(*tasks)
        return results


    def array2df(self, array, rate):
        """ Input phase!
            Transfer 1d array of time series data to 2d DataFrame, with a sampling rate of
            @param rate
        """


        array_idx = np.arange(array.shape[0])
        array_idx = np.array([idx for idx in array_idx if idx % rate == 0])
        array_with_index = np.c_[array_idx, array[array_idx]]
        # df = pl.DataFrame(array_with_index, {"Time": DataType.Int32, "Signal": DataType.Int16})
        df = pl.DataFrame(array_with_index, {"Time": None, "Signal": None})
        # print(f"df.shape {df.shape}")
        return df


    def back2origin(self, array, start_idx, end_idx, spinbox_value):
        """
        Output phase!
            @params:
            array: array to be transfered into original index
            start_idx, end_idx: must be divided by sample_rate. the both idx mean the 2 endpoints of an interval in a chunk
        """


        diff = spinbox_value * self.chunk_size
        original_start = start_idx + diff
        original_end = end_idx + diff

        array_idx_origin = np.arange(original_start, original_end)
        # array_with_index = np.c_[array_idx_origin, array[array_idx_origin]]
        array_origin = array[array_idx_origin]
        # df = pl.DataFrame(array_with_index, {"Time": DataType.Int32, "Signal": DataType.Int16, "path": DataType.String})


        return array_origin, original_start, original_end


    def origin2chunk(self, start_origin, end_origin):
        chunk_id_lower = int(start_origin / self.chunk_size)
        chunk_id_upper = int(end_origin / self.chunk_size)
        if chunk_id_lower != chunk_id_upper:
            print(f"chunk id conflict! lower={chunk_id_lower}, upper={chunk_id_upper}")
            return
        lower_idx_in_chunk = start_origin - self.chunk_size * chunk_id_lower
        upper_idx_in_chunk = end_origin - self.chunk_size * chunk_id_upper

        return chunk_id_lower, lower_idx_in_chunk, upper_idx_in_chunk


    def array2dict(self, array):
        pass


    def get_sample(self):
        return self.sample


    def convert_to_rate_times(num, rate, start=True):
        num = int(num)
        if num % rate != 0:
            if not start :
                # upper limit of an interval
                num += rate - (num % rate)
            else:
                # lower limit of an interval
                num -= (num % rate)
        return num


class AnalyzerBase():
    def __init__(self) -> None:
        self.file_path = None
        self.ch0 = None
        self.ch1 = None
        self.fgain0 = None
        self.fgain1 = None

    def _int_to_volt(self, ch: np.ndarray, gain: Union[float, None]) -> np.ndarray:
        """
        Scale an array of integers from the ADC to volts, given the gain.

        Parameters:
        -----------
        ch : np.ndarray
            The array of integers to scale.
        fgain : Union[float, None]
            The gain, in db, to use for scaling.

        Returns:
        --------
        np.ndarray
            The scaled array, in volts.
        """


        if ch is not None and gain is not None:
            return ch.astype(np.float64) / 2**15 * (40 / 10**(gain / 20))
        return ch

    def get_ch(self, ch: int = 0, scale: bool = False, samples: Optional[int] = None) -> np.ndarray:
        """
        Get the data for a given channel.

        Parameters:
        -----------
        ch : int
            The channel to read. Must be 0 or 1.
        scale : bool
            If True, scale the data to volts using the gain.
        samples : Optional[int]
            The number of samples to return. If None, the entire dataset is returned.

        Returns:
        --------
        np.ndarray
            The data for the given channel.
        """


        if ch not in [0, 1]:
            raise ValueError("Channel must be 0 or 1.")

        if samples is not None and not isinstance(samples, int):
            raise TypeError(f"samples must be an int or None, got {type(samples).__name__}.")

        data = self.ch0 if ch == 0 else self.ch1
        if data is None:
            raise ValueError(f"Channel {ch} data is not available.")

        fgain = self.fgain0 if ch == 0 else self.fgain1
        if fgain is None:
            raise ValueError(f"Channel {ch} fgain is not available.")

        data = data[:samples] if samples else data
        if scale:
            data = self._int_to_volt(data, fgain)

        return data

    def get_status(self) -> dict:
        return {
            'ch0': self.ch0,
            'ch1': self.ch1,
            'fgain0': self.fgain0,
            'fgain1': self.fgain1,
        }


class Npz(AnalyzerBase):
    def __init__(self, file_path: str) -> None:
        """
        Initialize an Analyzer recording from a .npz file.

        Parameters:
        -----------
        file_path : str
            The path to the .npz file to read.

        Returns:
        --------
        None
        """


        self.file_path = file_path
        self._load_data(self.file_path)


    def _load_data(self, file_path: str):
        """
        Load data from a given .npz file able to handle case with and without "adc_data".

        Parameters:
        -----------
        file_path : str
            The path to the .npz file to read.

        Returns:
        --------
        None
        """


        data = np.load(self.file_path)

        if 'adc_data' in data:
            adc_data = data['adc_data']
            self.ch0 = adc_data['ch0'] if 'ch0' in adc_data.dtype.names else None
            self.ch1 = adc_data['ch1'] if 'ch1' in adc_data.dtype.names else None
            self.fgain0 = data['fgain0'] if self.ch0 is not None else None
            self.fgain1 = data['fgain1'] if self.ch1 is not None else None
        else:
            self.ch0 = data['ina'] if 'ina' in data else None
            self.ch1 = data['inb'] if 'inb' in data else None
            self.fgain0 = data['fgain0'] if self.ch0 is not None else None
            self.fgain1 = data['fgain1'] if self.ch1 is not None else None


class NpyNpz(AnalyzerBase):
    def __init__(self, file_path: str) -> None:
        """
        Initialize an Analyzer recording from the data file.

        Parameters:
        -----------
        file_path : str
            The path to the .npy file to read.

        Returns:
        --------
        None
        """


        self.file_path = file_path
        self._load_data(file_path)


    def _load_data(self, file_path: str):
        """
        Load data from a given .npy and its corresponding *.npz settings file.

        Parameters:
        -----------
        file_path : str
            The path to the .npy file to read.

        Returns:
        --------
        None
        """


        data_path = Path(file_path)

        if data_path.suffix != '.npy':
            raise ValueError("File must be *.npy")

        if not data_path.exists():
            raise FileNotFoundError(f"The file {data_path} does not exist.")

        # Determine the corresponding settings file
        search_pattern = f"{data_path.stem.split('_och')[0]}*.npz"  # For "meas_recid1_ich1_traidd_och0_ch1", split('_ch') will produce a list: ["meas_recid1_ich1_traidd_och0", "1"]
        settings_path = list(data_path.parent.glob(search_pattern))[0]

        if not settings_path:
            raise FileNotFoundError(f"Corresponding file not found for pattern: {search_pattern}")

        settings = np.load(settings_path)

        self.ch0 = self._try_loading_channel(data_path, 'ch0')
        self.ch1 = self._try_loading_channel(data_path, 'ch1')
        self.fgain0 = settings['fgain0'] if self.ch0 is not None else None
        self.fgain1 = settings['fgain1'] if self.ch1 is not None else None

    def _try_loading_channel(self, data_path: Path, ch: Literal['ch0', 'ch1']):
        new_stem = data_path.stem[:-3] + ch
        channel_path = data_path.with_name(new_stem + data_path.suffix)

        if channel_path.exists():
            return np.load(channel_path, allow_pickle=True, mmap_mode='r')
        else:
            return None