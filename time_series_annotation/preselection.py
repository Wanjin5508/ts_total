import numpy as np
import polars as pl
import typing
from scipy import stats


class PreSelector:
    def __init__(self, dataframe, distance, segment_length=1000, height=100):
        """
        :param:
        :param:
        """
        self.segment_length = segment_length
        self.distance = distance
        self.height = height
        self.dataframe: pl.DataFrame = dataframe
        self.signal_array = self.dataframe["Signal"].to_numpy()
        self.n_segments = len(self.signal_array) // segment_length

        self.positive_df = self.get_positive_value()["Signal"]

        self.q1 = np.percentile(self.positive_df, 25)
        self.q3 = np.percentile(self.positive_df, 75)

    def build_intervals(self, recid, peak_idx_list, range_left=350, range_right=950):
        peak_idx = TimeSeriesPlot.peaks_idx_list_chunk

        intervals = []
        radius = self.segment_length // 2
        for peak_idx in peak_idx_list[recid]:
            start = peak_idx - range_left
            end = peak_idx + range_right
            intervals.append((start, end))
        return intervals

    def detect_abnormal_intervals(self) -> list:
        """
        return a list of tuple:
        (start_idx, end_idx) -> both are index in one chunk (aka in the plot)
        """
        if self.n_segments < 3:
            return []

        amplitudes = []
        avg_list = []
        for i in range(self.n_segments):
            segment = self.signal_array[i * self.segment_length:(i + 1) * self.segment_length]
            peak2peak = segment.max() - segment.min()

            avg_seg_pos = np.mean(segment[segment > 0])
            amplitudes.append(peak2peak)
            avg_list.append(avg_seg_pos)

        amplitudes = np.array(amplitudes)
        avg_list = np.array(avg_list)

        abnormal_intervals = []
        for i in range(1, self.n_segments - 1):
            current_amp = amplitudes[i]
            neighbor_amp = (amplitudes[i - 1] + amplitudes[i + 1]) / 2
            curr_avg = avg_list[i]
            if current_amp > self.height * neighbor_amp or curr_avg > self.q1:
                start_idx = i * self.segment_length
                end_idx = (i + 1) * self.segment_length - 1
                abnormal_intervals.append((start_idx, end_idx))

        return abnormal_intervals

    def get_positive_value(self):
        """
        :param: dataframe comes from timeseriesplot
        """
        positive_value_df = self.dataframe.filter(pl.col("Signal") > 0)
        # positive_value_df = self.dataframe.with_columns([
        #     pl.col("Signal").abs().alias("Signal")
        # ])
        return positive_value_df

    def calc_iqr(self):
        iqr = self.q3 - self.q1
        print(f'iqr : {iqr}')
        return self.q3 - self.q1
