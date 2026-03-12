import sys
import copy
import bisect

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from collections import defaultdict
from typing import DefaultDict, Dict, List
from scipy.signal import hilbert, find_peaks
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

from dataset_async import Dataset
from comment_rectangle import CommentLinearRegionItem
from preselection import PreSelector
from color_scheme import *

class TimeSeriesPlot(pg.PlotWidget):
    # define a signal to send length to mainwindow
    line_region_item_list_update = pyqtSignal(int)
    get_offset_signal = pyqtSignal(tuple)

    def __init__(self, parent=None, dataframe=None):
        super(TimeSeriesPlot, self).__init__(parent)
        self.setBackground('w')
        self.addLegend()
        # self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.drawing_item = None
        # self.setInteractive(True)
        pg.setConfigOptions(antialias=True)

        self.text = pg.TextItem('', anchor=(0,1), color=(0,0,0))
        self.text.setZValue(50)
        self.addItem(self.text, ignoreBounds=True)
        pen = pg.mkPen(color=(0, 0, 0))
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self.crosshair_v.setZValue(50)
        self.crosshair_h.setZValue(50)
        self.addItem(self.crosshair_v, ignoreBounds=True)
        self.addItem(self.crosshair_h, ignoreBounds=True)

        self.left_plotId_sampleIdx_tuple = None
        self.right_plotId_sampleIdx_tuple = None
        self.offset_info = pg.TextItem(
            "\t\t\n"
            "x1: (0, 0)\n"
            "x2: (0, 0)\n"
            "Δt: 0 ns\n",
            color=(20, 20, 20), anchor=(1, 1)
        )
        self.addItem(self.offset_info)
        self.view_box = self.getViewBox()
        self.view_box.sigRangeChanged.connect(self.update_offset_info_position)

        self.offset_indicator_item = None
        self.offset_indicator_item_limit_line_0 = None
        self.offset_indicator_item_limit_line_1 = None
        self.rate = 0
        self.chunk_size = 0

        self.addLine(y=0, pen=pg.mkPen('r', width=3))

        self.dataframe = dataframe

        self.start_pos = None
        self.x_offset_list = [0 for _ in range(len(self.dataframe))]

        self.plot_data_item_list = [idx for idx in range(len(self.dataframe))]
        self.line_envelope_list = copy.deepcopy(self.plot_data_item_list)
        self.line_ssm_envelope_list = copy.deepcopy(self.plot_data_item_list)
        self.line_rms_envelope_list = copy.deepcopy(self.plot_data_item_list)
        self.peaks_list = copy.deepcopy(self.plot_data_item_list)

        # to store the idx of peaks in a chunk
        self.peaks_idx_list_chunk = copy.deepcopy(self.plot_data_item_list)

        self.qt_checkboxes = [True for _ in Dataset.file_path]  # to receive state of each checkboxes of MainWindow
        self.offset_mode_checkbox = False

        self.plot_main()

        self.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.scene().sigMouseClicked.connect(self.unpick_selected_interval)
        self.sigRangeChanged.connect(self.updateAlpha)
        self.scene().sigMouseMoved.connect(self.onMouseMoved)

        self.selected_interval = None  # as a pointer, which point to an interval
        self.interval_range = (0.0, 0.0)  # range in the plot, not the index in df --> need to adjust rate
        self.interval_range_list = []

        self.signal_array = [[] for idx in range(len(dataframe))]  # to store peaks' coordinates
        self.pre_intervals = [[] for idx in range(len(dataframe))]
        self.picked_ir_items = []  # to store multiple selected ir items from preselected and manual
        self.te_export_rects = copy.deepcopy(self.pre_intervals)  # for export
        self.line_region_item_list = copy.deepcopy(self.pre_intervals)  # -> interval list, can be pointed by selected_interval

        self.te_export_rects_dict: DefaultDict[int, DefaultDict[int, List[CommentLinearRegionItem]]] = defaultdict(lambda: defaultdict(list))
        self.auto_pre_recid = 0
        self.preselector = None


    def get_new_chunk(self, tuple_of_dataframe: tuple) -> None:
        """
        This method can load a tuple of dataframes in the GUI and redraw pyqtgraph in the main window using new dataframes.

        :param tuple_of_dataframe: has the same size as the number of npy-files
        """
        self.dataframe = tuple_of_dataframe
        self.peaks_list = [idx for idx in range(len(self.dataframe))]
        self.peaks_idx_list_chunk = copy.deepcopy(self.peaks_list)

        self.update_plot()
        self.update_plot(20)


    def show_peaks(self, is_checked) -> None:
        print(f"show peaks is checked: {is_checked}")


    def remove_all_peaks_in_new_chunk(self) -> None:
        """Remove all peaks (circles) on the plot"""
        for peak_data_item in self.peaks_list:
            self.getPlotItem().removeItem(peak_data_item)
        self.peaks_list.pop()

        self.peaks_idx_list_chunk = [idx for idx in range(len(self.dataframe))]

    def remove_all_envelope_in_new_chunk(self) -> None:
        """Remove all envelopes from plot"""
        # for envelope_data_item in self.line_envelope_list:
        #     self.getPlotItem().removeItem(envelope_data_item)

        for envelope_idx in range(len(self.line_envelope_list)):
            self.getPlotItem().removeItem(self.line_envelope_list[envelope_idx])
            self.getPlotItem().removeItem(self.line_lsm_envelope_list[envelope_idx])
            self.getPlotItem().removeItem(self.line_ssm_envelope_list[envelope_idx])

        self.update()
        self.line_envelope_list = [idx for idx in range(len(self.dataframe))]
        self.line_lsm_envelope_list = copy.deepcopy(self.line_envelope_list)
        self.line_ssm_envelope_list = copy.deepcopy(self.line_envelope_list)
        self.line_rms_envelope_list = copy.deepcopy(self.line_envelope_list)

    def plot_main(self) -> None:
        """The main plotting method, to plot the provided dataframes in the main window.
        But does not need to redraw a plot as new chunks come. Use update_plot() instead.
        """
        for idx, df in enumerate(self.dataframe):
            pen = pg.mkPen(color=COLORS[idx])
            dataline = self.plot(df["Time"], df["Signal"], pen=pen, name=Dataset.file_path[idx])
            # return: <class 'pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem'>
            dataline.setAlpha(alpha=0.3, auto=False)
            dataline.setZValue(40 - idx)
            dataline.setDownsampling(ds=5, auto=True, method='peak')

            # set the curve (plotdataitem) to be clickable
            dataline.setCurveClickable(True, width=3)
            dataline.sigClicked.connect(self.on_curve_clicked)
            self.plot_data_item_list[idx] = dataline

        self.setLabel("left", "Signal")
        self.setLabel("bottom", "Time")

    def update_plot(self, alpha=None) -> None:
        """
        For good performance when drawing the time series.

        :param alpha: to reset PlotDataItem's transparency for a good visual effect,
                      by default is None so we can call it without param to only update data.
        """
        # use setData() rather than run plot_main again
        for idx, df in enumerate(self.dataframe):
            self.plot_data_item_list[idx].setData(df["Time"] + self.x_offset_list[idx], df["Signal"])

        for chunk_id, exported_rect_list in self.te_export_rects_dict.items():
            # add the exported grey intervals to Plot again
            for exported_rect in exported_rect_list:
                if exported_rect in self.plotItem.items:
                    continue
                if chunk_id == Dataset.chunk_idx:
                    self.addItem(exported_rect)
                else:
                    self.removeItem(exported_rect)
            # for chunk_idx in self.te_export_rects_dict[idx].keys():
            #     if chunk_idx == Dataset.chunk_idx:
            #         exported_rect.setVisible(True)
            #     else:
            #         exported_rect.setVisible(False)

        if alpha:
            self.plot_data_item_list[idx].setAlpha(alpha=alpha, auto=True)

    def shift_plot(self, recid, direction, range):
        range_with_direction = int(direction * range)
        self.x_offset_list[recid] += range_with_direction
        self.plot_data_item_list[recid].setData(
            self.dataframe[recid]["Time"] + self.x_offset_list[recid],
            self.dataframe[recid]["Signal"]
        )

    def mouse_clicked(self, event):
        """
        Slot, triggered when mouse is clicked.
        Get the position of mouse and select a region of interest via 2 single clicks that build an interval.
        """
        # if event.button() == 2:
        # handle the right click event, in order to send a signal to mw and update line edit
        #     print("!!!")
        if event.button() != 1:
            event.ignore()
            return

        pos = self.plotItem.vb.mapSceneToView(event.scenePos()).x()
        recid = self.get_auto_pre_recid()
        to_detect = self.pre_intervals[recid] + self.line_region_item_list[recid]

        if self.offset_mode_checkbox:
            # record the selected offset and emit it to mw
            offset = self.get_offset()
            self.x_offset_list[recid] = offset
            if offset is not None:
                self.get_offset_signal.emit(offset)
                self.start_pos = None
                x_coord_pair = offset
                if self.offset_indicator_item is not None:
                    self.removeItem(self.offset_indicator_item)
                    self.offset_indicator_item = None
                self.offset_indicator_item = pg.LinearRegionItem(x_coord_pair, movable=False, pen=pg.mkPen(color=(0, 255, 0)))
                self.addItem(self.offset_indicator_item)
                time_pair = self.calc_time_pair(x_coord_pair)
                delta_t = x_coord_pair[0] - x_coord_pair[1]  # according to formel, always use t0 - t1
                delta_x = (0, 0)
                self.offset_info.setText(
                    f"x1: ({int(x_coord_pair[0])})\n"
                    f"x2: ({int(x_coord_pair[1])})\n"
                    f"Δx: {delta_x}\n"
                    f"Δt: {delta_t} ns\n"
                )
            return

                # check whether pos is in one of the LinearRegionItems
        for line_region_item in to_detect:
            min_x, max_x = line_region_item.getRegion()
            if min_x <= pos <= max_x and not line_region_item.is_exported:
                self.select_region(line_region_item)
                return

        if self.start_pos is None:
            self.start_pos = pos
            return

        end_pos = pos
        if np.abs(self.start_pos - end_pos) < 10:
            return

        interval = CommentLinearRegionItem(
            [self.start_pos, end_pos],
            recid=recid,
            clipItem=self.plot_data_item_list[recid],
            color_list=[SELECTED_COLORS[recid], PICKED_COLORS[recid]]
        )
        interval.setZValue(10)
        self.addItem(interval)
        self.line_region_item_list[recid].append(interval)

        # send signal to mainwindow
        self.line_region_item_list_update.emit(
            len(self.line_region_item_list[self.get_auto_pre_recid()])
        )
        self.selected_interval = interval
        self.start_pos = None
        print(f'selected interval: {interval.getRegion()}')
        print(f'selected interval len: {interval.getRegion()[1] - interval.getRegion()[0]}')

    def on_curve_clicked(self, self_plot, event):
        if self.left_plotId_sampleIdx_tuple and self.right_plotId_sampleIdx_tuple:
            self.left_plotId_sampleIdx_tuple = None
            self.right_plotId_sampleIdx_tuple = None
            self.removeItem(self.offset_indicator_item_limit_line_0)
            self.removeItem(self.offset_indicator_item_limit_line_1)
            self.offset_indicator_item_limit_line_0 = None
            self.offset_indicator_item_limit_line_1 = None

        mousePoint = event.scenePos()
        x_click = self.plotItem.vb.mapSceneToView(mousePoint).x()
        x_data = self_plot.getData()[0]

        lower_limit = bisect.bisect_left(x_data, x_click - 10)
        upper_limit = bisect.bisect_right(x_data, x_click + 10)
        x_range = x_data[lower_limit:upper_limit]
        print(f"xrange = {x_range}")

        distances = np.abs(x_range - x_click)
        closest_index = np.argmin(distances)

        print(f"用户点击了折线：{plot.name()}, 坐标位置：{x_click}, 最近索引对应的横坐标：{x_range[closest_index]}")
        print(f"idx of plot: {Dataset.file_path.index(plot.name())}")

        plot_idx = Dataset.file_path.index(plot.name())
        if plot_idx % 2 == 0 and self.left_plotId_sampleIdx_tuple is None:
            self.left_plotId_sampleIdx_tuple = (plot_idx, x_range[closest_index])
            self.offset_indicator_item_limit_line_0 = pg.InfiniteLine(
                x_range[closest_index], angle=90, pen=pg.mkPen(color=(0, 255, 0))
            )
            self.addItem(self.offset_indicator_item_limit_line_0)

        if plot_idx % 2 == 1 and self.right_plotId_sampleIdx_tuple is None:
            self.right_plotId_sampleIdx_tuple = (plot_idx, x_range[closest_index])
            self.offset_indicator_item_limit_line_1 = pg.InfiniteLine(
                x_range[closest_index], angle=90, pen=pg.mkPen(color=(0, 255, 0))
            )
            self.addItem(self.offset_indicator_item_limit_line_1)


    def get_offset(self):
        # if self.start_pos is None:
        #     self.start_pos = pos
        #     return None
        # else:
        #     end_pos = pos
        #     if end_pos <= self.start_pos:
        #         offset = (round(end_pos), round(self.start_pos))
        #     else:
        #         offset = (round(self.start_pos), round(end_pos))
        #     return offset

        if self.left_plotId_sampleIdx_tuple is None or self.right_plotId_sampleIdx_tuple is None:
            return None

        if self.left_plotId_sampleIdx_tuple[0] + 1 != self.right_plotId_sampleIdx_tuple[0]:
            self.left_plotId_sampleIdx_tuple = None
            self.right_plotId_sampleIdx_tuple = None
            self.removeItem(self.offset_indicator_item_limit_line_0)
            self.removeItem(self.offset_indicator_item_limit_line_1)
            self.offset_indicator_item_limit_line_0 = None
            self.offset_indicator_item_limit_line_1 = None
            print("wrong streams selected!")
            return None 
        offset = (self.left_plotId_sampleIdx_tuple[1], self.right_plotId_sampleIdx_tuple[1])
        return offset
    
        def calc_time_pair(self, x_coord_pair) -> tuple:
        # print(f'compare_abs_signal x0: {self.compare_abs_signal(x_coord_pair[0])}')
        # print(f'compare_abs_signal x1: {self.compare_abs_signal(x_coord_pair[1])}')
            t0 = 16 * x_coord_pair[0]   # just use ns
            t1 = 16 * x_coord_pair[1]
        return t0, t1

    def compare_abs_signal(self, x) -> int:
        if sum(self.qt_checkboxes) != 2:
            return
        # recid 0
        idx_rec_left = self.qt_checkboxes.index(True)
        # recid 1
        idx_rec_right = idx_rec_left + 1

        signal_left = self.dataframe[idx_rec_left]
        signal_right = self.dataframe[idx_rec_right]
        # print(f'signal_left[x + self.x_offset_list[idx_rec_left]]: {signal_left[x + self.x_offset_list[idx_rec_left]]}')
        # print(f'signal_right[x + self.x_offset_list[idx_rec_right]]: {signal_right[x + self.x_offset_list[idx_rec_right]]}')

        upper_multiple = Dataset.chunk_size * Dataset.chunk_idx + Dataset.rate
        lower_multiple = Dataset.chunk_size * Dataset.chunk_idx * np.floor(x/Dataset.rate) * Dataset.rate
        x = upper_multiple

        if np.abs(signal_left["Signal"][int(x + self.x_offset_list[idx_rec_left])]) > np.abs(signal_right["Signal"][int(x + self.x_offset_list[idx_rec_right])]):
            return idx_rec_left
        else:
            return idx_rec_right

    def unpick_selected_interval(self, event):
        """
        undo the picked interval (orange), and turn it to selected state (yellow)
        """
        if event.button() != 2 or self.selected_interval is None:
            return
        pos = self.plotItem.vb.mapSceneToView(event.scenePos()).x()
        x_lower, x_upper = self.selected_interval.getRegion()
        if x_lower <= pos <= x_upper:
            # self.selected_interval.setBrush(pg.mkBrush(255, 255, 0, 12))
            self.selected_interval.on_right_click_unpick()
            if self.selected_interval in self.picked_ir_items:
                self.picked_ir_items.remove(self.selected_interval)
                
            self.selected_interval = None

    def update_offset_info_position(self):
        view_box = self.getViewBox()
        view_rect = view_box.viewRect()
        self.offset_info.setPos(view_rect.topRight())

    @pyqtSlot()  # use decorator for better performance.
    def updateAlpha(self):
        # get the range
        view_range = self.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        # print(f'x range {x_range}')

        # set alpha according to the range of view
        alpha = max(30, min(255, int(255 * (300 / x_range))))
        self.update_plot(alpha=alpha)
        # print(f'@pyqtSlot alpha = {alpha}')

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y()
            x_range = self.getViewBox().viewRange()[0]
            current_x_min, current_x_max = x_range
            shift_amount = -10
            new_x_min = current_x_min + shift_amount * delta
            new_x_max = current_x_max + shift_amount * delta
            self.setRange(xRange=(new_x_min, new_x_max), padding=0)
            self.getViewBox().setRange(xRange=(new_x_min, new_x_max))
            event.accept()
        else:
            super().wheelEvent(event)

    def onMouseMoved(self, self_, event):
        """
        Automated update the textItem, which shows coordinates of the cursor.
        """
        pos = event
        if self.sceneBoundingRect().contains(pos):
            mousePoint = self.getPlotItem().vb.mapSceneToView(pos)
            self.text.setText(f"x: {mousePoint.x():.1f}, y: {mousePoint.y():.1f}")
            self.text.setPos(mousePoint.x(), mousePoint.y())
            self.crosshair_v.setPos(mousePoint.x())
            self.crosshair_h.setPos(mousePoint.y())

    def select_region(self, line_region_item: CommentLinearRegionItem) -> None:
        # # reset the already picked interval in yellow
        # if self.selected_interval is not None:
        #     self.selected_interval.setBrush(pg.mkBrush(255, 255, 0, 128))  # "yellow"

        # set the new picked interval in orange
        self.selected_interval = line_region_item
        self.selected_interval.setBrush(pg.mkBrush(255, 128, 0, 128))  # "orange"
        line_region_item.clicked.connect(self.on_region_clicked)
        self.start_pos = None
        self.selected_interval = line_region_item

    def on_region_clicked(self, region: CommentLinearRegionItem) -> None:
        if region.is_exported:
            region.redo_annotate_rect()
            self.line_region_item_list[self.get_auto_pre_recid()].append(region)
            return

        region.set_selected(not region.is_selected)
        if region.is_selected:
            self.picked_lr_items.append(region)
        else:
            self.picked_lr_items.remove(region)
            if self.selected_interval == region:
                self.selected_interval = None
        # print(f"self.picked_lr_items len = {len(self.picked_lr_items)}")
        # print(f"({region in self.line_region_item_list[self.get_auto_pre_recid()]}, region in self.pre_intervals[self.get_auto_pre_recid()]})")


    def set_auto_pre_recid(self, index):
        self.auto_pre_recid = index

    def get_auto_pre_recid(self) -> int:
        return self.auto_pre_recid

    def toggle_line(self, line, is_visible: bool):
        try:
            line.setVisible(is_visible)
        except AttributeError as e:
            # print("no peaks to show or hide")
            return

    def preselection(self, recid: int = 0, distance: int = 1000, height: int = 100, show_peaks: bool = True,
                     window_size=300, threshold=100, envelope_id=0):
        self.preselector = PreSelector(self.dataframe[recid], distance, height)

        # initialize preselected intervals
        self.pre_intervals[recid] = []
        self.plot_envelope(recid, height=height, distance=distance, show_peaks=show_peaks,
                           window_size=window_size, threshold=threshold, envelope_id=envelope_id)
        self.calc_peaks_in_chunk(height=height, distance=distance)

        # generate (start, end) list for this recid
        intervals = self.preselector.build_intervals(recid, self.peaks_idx_list_chunk)

        for start, end in intervals:
            # build new linear region items
            rect = CommentLinearRegionItem(
                [start + self.x_offset_list[recid], end + self.x_offset_list[recid]],
                clipItem=self.plot_data_item_list[self.get_auto_pre_recid()],
                is_preselected=True,
                recid=recid,
                color_list=[SELECTED_COLORS[recid], PICKED_COLORS[recid]]
            )
            if rect in self.te_export_rects_dict[recid][Dataset.chunk_idx]:
                continue
            rect.setZValue(10)
            self.pre_intervals[recid].append(rect)
            self.addItem(rect)
            rect.setVisible(self.qt_checkboxes[recid])

    def plot_envelope(self, recid, height=100, distance=100, show_peaks=True, window_size=100, threshold=100, envelope_id=0):
        signal_array = self.dataframe[recid].to_numpy()
        signal = signal_array[:, 1]
        t = signal_array[:, 0]

        pen = pg.mkPen(COLORS[-1])

        max_envelope, sum_square_envelop, rms_envelope = self.sliding_max_envelope(t, signal, window_size=window_size)
        reference_line_list = [max_envelope, sum_square_envelop, rms_envelope]

        peaks, _ = find_peaks(reference_line_list[envelope_id], height=height, distance=distance)  # signal

        sustained_spikes_list = self.detect_sustained_spikes(reference_line_list[envelope_id], threshold=threshold)  # max_envelope

        for start, end in sustained_spikes_list:
            peaks = [t[peaks] < start][t[peaks] > end]
            rect = CommentLinearRegionItem(
                [start + self.x_offset_list[recid], end + self.x_offset_list[recid]],
                clipItem=self.plot_data_item_list[recid],
                is_preselected=True,
                recid=recid,
                color_list=[SELECTED_COLORS[recid], PICKED_COLORS[recid]]
            )
            # print(f'rect.recid = {rect.recid}')
            rect.setZValue(10)
            self.pre_intervals[recid].append(rect)
            self.addItem(rect)
            rect.setVisible(self.qt_checkboxes[recid])

        temp_peaks = self.plot(t[peaks] + self.x_offset_list[recid],
                               reference_line_list[envelope_id][peaks],
                               symbol='o', pen=None, symbolBrush=COLORS[recid], symbolSize=10)
        temp_peaks.setVisible(show_peaks and self.qt_checkboxes[recid])

        temp_envelope = self.plot(t + self.x_offset_list[recid], max_envelope, pen=pen)
        temp_envelope_ssm = self.plot(t + self.x_offset_list[recid], sum_square_envelop,
                                      pen=pg.mkPen(color=(245, 0, 0, 128)))
        temp_envelope_rms = self.plot(t + self.x_offset_list[recid], rms_envelope, pen=pg.mkPen(color=(0, 245, 0, 128)))

        temp_envelope.setZValue(40)
        temp_envelope_ssm.setZValue(40)
        temp_envelope_rms.setZValue(40)

        temp_envelope.setVisible(show_peaks and self.qt_checkboxes[recid])
        temp_envelope_ssm.setVisible(show_peaks and self.qt_checkboxes[recid])
        temp_envelope_rms.setVisible(show_peaks and self.qt_checkboxes[recid])

        self.update()

        self.peaks_list[recid] = temp_peaks
        self.peaks_idx_list_chunk[recid] = t[peaks]
        self.line_envelope_list[recid] = temp_envelope
        self.line_ssm_envelope_list[recid] = temp_envelope_ssm
        self.line_rms_envelope_list[recid] = temp_envelope_rms

    def detect_sustained_spikes(self, time_array: np.ndarray, signal_envelope: np.ndarray,
                                threshold: float, min_len: float = 100):
        above_threshold = (signal_envelope > threshold)
        # print(above_threshold)
        regions = []
        start_idx = None

        for i, val in enumerate(above_threshold):
            if val:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    end_idx = i - 1
                    length = end_idx - start_idx + 1
                    if length > min_len:
                        regions.append((time_array[start_idx], time_array[end_idx]))
                    start_idx = None
                    continue

        if start_idx is not None:
            end_idx = len(above_threshold) - 1
            length = end_idx - start_idx + 1
            if length > min_len:
                regions.append((time_array[start_idx], time_array[end_idx]))
        return regions

    def sliding_max_envelope(self, time, signal, window_size=100):
        """
        use sliding window to calculate envelope

        param:
        - signal: the input time series
        - window_size: the size of window in the plot

        return:
        - smoothed envelope
        """

        # window size in UI = 5 * array index
        rate = time[1] - time[0]
        df_window = window_size // rate

        max_envelope = np.zeros_like(signal)
        sum_square_envelop = np.zeros_like(signal, dtype=np.int64)
        rms_envelope = np.zeros_like(signal)

        # replace negative value into the base of logarithms
        signal = np.where(signal < 0, 2, signal)
        signal = np.abs(signal)

        for i in range(0, len(signal)):
            # make sure index inside the boundaries
            start = max(0, i - df_window)
            end = min(len(signal), i + df_window + 1)

            sum_window_square = np.sum(np.square(signal[start:end]))

            max_envelope[i] = np.max(signal[start:end])
            sum_square_envelop[i] = np.sqrt(sum_window_square) ** 2
            rms_envelope[i] = np.sqrt(sum_window_square / df_window)

        return max_envelope, sum_square_envelop, rms_envelope

    def init_to_export_rects_dict(self):
        self.te_export_rects_dict = defaultdict(lambda: defaultdict(list))














