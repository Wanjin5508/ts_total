import pandas as pd  
import datetime 
import sys  


from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # --> is a QWidget 
from matplotlib.figure import Figure 
from matplotlib.widgets import Cursor, RangeSlider, Button 
from PyQt5.QtCore import Qt  
import matplotlib.dates as mdates 
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle 
from collections import defaultdict 
import sys
import pandas as pd
import numpy as np  
from PyQt5.QtWidgets import QSlider


from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, dataframe=None):
        fig = Figure()
        self.ax_main_0 = fig.add_subplot(321)  # 第一个主图
        self.ax_main_1 = fig.add_subplot(322)  # 第二个主图
        self.ax_zoom_0 = fig.add_subplot(325)  # 第一个放大展示图
        self.ax_zoom_1 = fig.add_subplot(326)  # 第二个放大展示图

        super(TimeSeriesPlot, self).__init__(fig)
        self.parent = parent
        self.dataframe = dataframe
        
        


        self.selected_regions_0 = []  # ax_main_0 的选中区间列表
        self.selected_regions_1 = []  # ax_main_1 的选中区间列表
        
        self.selected_region_0 = None
        self.selected_region_1 = None

        # 初始化时绘制全部数据
        self.ax_main_0.plot(self.dataframe['Time'], self.dataframe['Signal'], label='Device 1')
        self.ax_main_1.plot(self.dataframe['Time'], self.dataframe['Signal'], label='Device 2')

        self.ax_main_0.set_title('Main Plot 1')
        self.ax_main_1.set_title('Main Plot 2')

        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)

        # 使图例可见
        self.ax_main_0.legend()
        self.ax_main_1.legend()
        
        

        self.draw()  # 更新绘图区显示
        
        self.press = None 
        self.release = None 
        self.press_inaxes = None 
        self.current_span = None 
        
    

    def update_interval(self, plot_index, slider_type, value):
        selected_region = getattr(self, f'selected_region_{plot_index}', None)
        if selected_region:
            region, (start, end) = selected_region
            if slider_type == 'start':
                new_start = min(value, end - 1)  # 确保start小于end
                new_end = end
            else:
                new_start = start
                new_end = max(value, start + 1)  # 确保end大于start

            # 更新区间显示
            region.set_xy([[new_start, 0], [new_start, 1], [new_end, 1], [new_end, 0]])
            setattr(self, f'selected_region_{plot_index}', (region, (new_start, new_end)))
            self.plot_zoom(plot_index, new_start, new_end)


    def on_press(self, event):
        if event.inaxes not in [self.ax_main_0, self.ax_main_1]:
            return
        
        # if self.press_inaxes == self.ax_main_0:
        #     region = self.ax_main_0.axvspan(start_time, end_time, color='yellow', alpha=0.5, picker=True)
        #     self.selected_regions_0.append((region, (start_time, end_time)))
            
        self.press = event.xdata
        self.press_inaxes = event.inaxes

    def on_release(self, event):
        if self.press is None or event.inaxes != self.press_inaxes:
            return
        
        release = event.xdata
        start_time = min(self.press, release)
        end_time = max(self.press, release)
        
        # 根据选中的图区分处理
        if self.press_inaxes == self.ax_main_0:
            region = self.ax_main_0.axvspan(start_time, end_time, color='yellow', alpha=0.5, picker=True)
            self.selected_regions_0.append((region, (start_time, end_time)))
            self.plot_zoom(self.ax_zoom_0, start_time, end_time)
        elif self.press_inaxes == self.ax_main_1:
            region = self.ax_main_1.axvspan(start_time, end_time, color='green', alpha=0.5, picker=True)
            self.selected_regions_1.append((region, (start_time, end_time)))
            self.plot_zoom(self.ax_zoom_1, start_time, end_time)

        self.press = None  # 重置press位置
        self.press_inaxes = None  # 重置按下时的轴位置

    def plot_zoom(self, plot_index, start, end):
        ax = self.ax_zoom_0 if plot_index == 0 else self.ax_zoom_1
        ax.clear()
        mask = (self.dataframe['Time'] >= start) & (self.dataframe['Time'] <= end)
        ax.plot(self.dataframe['Time'][mask], self.dataframe['Signal'][mask])
        ax.set_title(f'放大视图: {start} 至 {end}')
        ax.grid(True)
        self.draw()

    # 其他相关方法继续在这里添加
