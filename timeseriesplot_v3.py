import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, dataframe=None):
        fig = Figure()
        self.ax_main_0 = fig.add_subplot(321)
        self.ax_main_1 = fig.add_subplot(322)
        self.ax_zoom_0 = fig.add_subplot(325)
        self.ax_zoom_1 = fig.add_subplot(326)

        super(TimeSeriesPlot, self).__init__(fig)
        self.parent = parent
        self.dataframe = dataframe

        self.selected_regions_0 = []
        self.selected_regions_1 = []
        self.selected_region = None  # 用来存储当前选中的区间

        self.ax_main_0.plot(self.dataframe['Time'], self.dataframe['Signal'])
        self.ax_main_1.plot(self.dataframe['Time'], self.dataframe['Signal'])

        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('pick_event', self.on_pick)

        # 创建一个水平滑块
        self.slider = QSlider(Qt.Horizontal, parent)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.dataframe['Time']) - 1)
        self.slider.valueChanged.connect(self.update_region_from_slider)

    def on_press(self, event):
        if event.inaxes not in [self.ax_main_0, self.ax_main_1]:
            return
        self.press = event.xdata
        self.press_inaxes = event.inaxes  # Ensure this line is here

    def on_release(self, event):
        if self.press is None or event.inaxes != self.press_inaxes:
            return
        release = event.xdata
        start_time = min(self.press, release)
        end_time = max(self.press, release)
        if event.inaxes == self.ax_main_0:
            region = self.ax_main_0.axvspan(start_time, end_time, color='yellow', alpha=0.5, picker=True)
            self.selected_regions_0.append((region, (start_time, end_time)))
        else:
            region = self.ax_main_1.axvspan(start_time, end_time, color='green', alpha=0.5, picker=True)
            self.selected_regions_1.append((region, (start_time, end_time)))

        self.selected_region = (region, (start_time, end_time))  # Store the selected region
        self.press = None
        self.press_inaxes = None  # Reset to avoid carrying over the value

    def on_pick(self, event):
        self.selected_region = next(((r, (start, end)) for r, (start, end) in self.selected_regions_0 if r == event.artist), None)
        if self.selected_region:
            _, (start, end) = self.selected_region
            self.slider.setValue(start)  # 这里可以调整为设置一个区间值，如果有对应的组件

    def update_region_from_slider(self, value):
        if self.selected_region:
            region, (start, _) = self.selected_region
            new_end = value
            region.set_xy([[start, 0], [start, 1], [new_end, 1], [new_end, 0]])
            self.selected_region = (region, (start, new_end))
            self.draw()

# 其他代码，例如创建 QApplication, MainWindow 等，需要根据实际情况添加。
