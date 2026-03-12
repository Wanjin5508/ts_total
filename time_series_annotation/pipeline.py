import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from paths import *
from pathlib import Path


def load_data(folder_name: str, filter_on_t0: bool):
    folder_name_obj = Path(folder_name)
    if filter_on_t0:
        path = DATA_ROOT / folder_name_obj / T0_DATA
    else:
        path = DATA_ROOT / folder_name_obj / T1_DATA

    return np.load(path)


def lowpass_filter(signal_data, fs, fc):
    """
    Apply a low-pass Butterworth filter to the given signal data.

    Parameters:
    - signal_data: The input signal to be filtered.
    - fs: The sampling frequency (Hz).
    - fc: The cut-off frequency (Hz).

    Returns:
    - filtered_signal: The filtered signal.
    """

    # Normalize the cut-off frequency
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = fc / nyquist

    # Design the Butterworth low-pass filter
    b, a = signal.butter(8, normal_cutoff, btype='low', analog=False)

    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal.astype(np.float32)


def save_filtered_data(folder_name: str, filter_on_t0: bool = True, filtered_data: np.ndarray = None):
    folder_name_obj = Path(folder_name)
    if filter_on_t0:
        path = DATA_ROOT / folder_name_obj / T0_FILTERED_DATA
    else:
        path = DATA_ROOT / folder_name_obj / T1_FILTERED_DATA

    np.save(path, filtered_data)


def plot_data_original_filtered(original_data, filtered_data):
    plt.figure()
    plt.plot(original_data[:100000])
    plt.plot(filtered_data[:100000].astype(np.int16))
    plt.show()


def lowpass_pipeline(folder_name: str, fs: int, fc: int, filter_on_t0: bool = True):
    original_data = load_data(folder_name, filter_on_t0)
    filtered_data = lowpass_filter(original_data, fs, fc)
    save_filtered_data(folder_name, filter_on_t0, filtered_data)
    plot_data_original_filtered(original_data, filtered_data)
