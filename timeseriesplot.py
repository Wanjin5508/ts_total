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

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from timeseriesplot import *

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, dataframe=None ):
        self.fig = Figure(figsize=(20, 12))
        
        super(TimeSeriesPlot, self).__init__(self.fig)
        
        self.ax_main_0 = self.fig.add_subplot(321)
        self.ax_main_1 = self.fig.add_subplot(322)
        
        self.ax_zoom_0 = self.fig.add_subplot(323)
        self.ax_zoom_1 = self.fig.add_subplot(324)
        
        self.fig.tight_layout()
        
        self.ax_main_0.set_xlim(0, 20)
        self.ax_main_0.set_ylim(0, 40)
        
        self.ax_main_1.set_xlim(21, 40)
        self.ax_main_1.set_ylim(0, 40)
        
        self.dataframe = dataframe 
        self.plot_main()
        
        
        self.press = None 
        self.release = None 
        self.press_inaxes = None 
        self.current_span = None 
        
        self.ax_slider_0 = self.fig.add_subplot(325)
        self.slider_0 = RangeSlider(self.ax_slider_0, 'Threshold_0', self.dataframe['Time'].min(), self.dataframe['Time'].max())
        
        self.ax_slider_1 = self.fig.add_subplot(326)
        self.slider_1 = RangeSlider(self.ax_slider_1, 'Threshold_1', self.dataframe['Time'].min(), self.dataframe['Time'].max())
        
        # a dictionary to used to change interval limit
        self.limit_line = defaultdict(list)
        self.lower_limit_line_0 = None
        self.upper_limit_line_0 = None
        self.lower_limit_line_1 = None
        self.upper_limit_line_1 = None
        
        self.slider_0.on_changed(self.update_main_0)
        self.slider_1.on_changed(self.update_main_1)
        
        self.selected_intervals = []
        # todo: 选中区间列表
        self.selected_regions_0 = []  # ax_main_0的选中区间列表
        self.selected_regions_1 = []  # ax_main_1的选中区间列表
        
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('button_press_event', self.onselect)
        
        
        
        
    def plot_main(self):
        self.ax_main_0.clear()
        self.ax_main_0.plot(self.dataframe['Time'], self.dataframe['Signal'], label='Signal Strength')
        self.ax_main_0.set_title('Signal Strength 0')
        self.ax_main_0.set_xlabel('Time')
        self.ax_main_0.set_ylabel('Signal Strength')
        self.ax_main_0.legend()
        self.ax_main_0.grid(visible=True)
        
        self.ax_main_1.clear()
        self.ax_main_1.plot(self.dataframe['Time'], self.dataframe['Signal'], label='Signal Strength')
        self.ax_main_1.set_title('Signal Strength 1')
        self.ax_main_1.set_xlabel('Time')
        self.ax_main_1.set_ylabel('Signal Strength')
        self.ax_main_1.legend()
        self.ax_main_1.grid(visible=True)
        
        
        
        cursor_0 = Cursor(self.ax_main_0, useblit=True, color='red', linewidth=2)
        cursor_1 = Cursor(self.ax_main_1, useblit=True, color='red', linewidth=2)
        
        def onclick(event):
            cursor_0.onmove(event)
            cursor_1.onmove(event)
            
        self.mpl_connect('button_press_event', onclick)
        self.draw()
    
    def plot_zoom_0(self, start_time, end_time):
        self.ax_zoom_0.clear()
        start_datetime = start_time
        end_datetime = end_time
        
        mask = (self.dataframe['Time'] >= start_datetime) & (self.dataframe['Time'] <= end_datetime)
        zoom_data = self.dataframe.loc[mask]
        
        if not zoom_data.empty:
            self.ax_zoom_0.plot(zoom_data['Time'], zoom_data['Signal'], color='orange', label='Signal Strength')
            self.ax_zoom_0.set_title('Signal Strength 0')
            self.ax_zoom_0.set_xlabel('Time')
            self.ax_zoom_0.set_ylabel('Signal Strength')
            self.ax_zoom_0.legend()
            self.ax_zoom_0.grid(visible=True)
            
        self.draw()
        
    
    def plot_zoom_1(self, start_time, end_time):
        self.ax_zoom_1.clear()
        
        start_datetime = start_time
        end_datetime = end_time
        
        mask = (self.dataframe['Time'] >= start_datetime) & (self.dataframe['Time'] <= end_datetime)
        zoom_data = self.dataframe.loc[mask]
        
        if not zoom_data.empty:
            self.ax_zoom_1.plot(zoom_data['Time'], zoom_data['Signal'], color='orange', label='Signal Strength')
            self.ax_zoom_1.set_title('Signal Strength 1')
            self.ax_zoom_1.set_xlabel('Time')
            self.ax_zoom_1.set_ylabel('Signal Strength')
            self.ax_zoom_1.legend()
            self.ax_zoom_1.grid(visible=True)
            
        self.draw()
        
        
    def plot_zoom(self, ax, start, end):
        ax.clear()  # 清除当前的轴，准备新的绘图
        mask = (self.dataframe['Time'] >= start) & (self.dataframe['Time'] <= end)
        ax.plot(self.dataframe['Time'][mask], self.dataframe['Signal'][mask])
        ax.set_title(f'Zoomed view from {start} to {end}')
        ax.grid(True)  # 添加网格线以便更好的视觉对比
        self.draw()  # 重新绘制整个图形，以更新显示

        
    # def on_press(self, event):
    #     if event.inaxes != self.ax_main_0 and event.inaxes != self.ax_main_1:
    #         return
        
    #     self.press = event.xdata
    #     self.press_inaxes = event.inaxes
        
    #     self.lower_limit_line_0 = self.ax_main_0.axvline(x=self.press, ymin=0, ymax=1, color='red', linestyle='--')
        
    # def on_release(self, event):
    #     if self.press is None or (event.inaxes != self.ax_main_0 and event.inaxes != self.ax_main_1):
    #         return
    #     if self.press_inaxes != event.inaxes:
    #         return
        
    #     self.release = event.xdata
    #     start_time = min(self.press, self.release )
    #     end_time = max(self.press, self.release )
        
    #     if self.press_inaxes == self.ax_main_0:
    #         self.current_span = self.ax_main_0.axvspan(self.press, self.release, color='yellow', alpha=0.2)
    #         self.plot_zoom_0(start_time, end_time)
    #         self.upper_limit_line_0 = self.ax_main_0.axvline(self.release, color='k')
            
    #     else:
    #         self.current = self.ax_main_1.axvspan(self.press, self.release, color='green', alpha=0.2)
    #         self.plot_zoom_1(start_time, end_time)
            
    #     self.press = None 
    #     self.release = None 
    
    def on_press(self, event):
        if event.inaxes not in [self.ax_main_0, self.ax_main_1]:
            return
        
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
            region = self.ax_main_0.axvspan(start_time, end_time, color='yellow', alpha=0.5)
            self.selected_regions_0.append((region, (start_time, end_time)))
            self.plot_zoom(self.ax_zoom_0, start_time, end_time)
        elif self.press_inaxes == self.ax_main_1:
            region = self.ax_main_1.axvspan(start_time, end_time, color='green', alpha=0.5)
            self.selected_regions_1.append((region, (start_time, end_time)))
            self.plot_zoom(self.ax_zoom_1, start_time, end_time)
        
        self.press = None  # 重置press位置
        
    def onselect(self, event):
        if event.inaxes == self.ax_main_0:
            xdata = event.xdata  
            if event.button == 1:
                self.slider_0.set_val((xdata, self.slider_0.val[1]))
            elif event.button == 3:
                self.slider_0.set_val((self.slider_0.val[0], xdata))
        
        
    def update_main_0(self, val):
        self.lower_limit_line_0.set_xdata(val[0])
        self.upper_limit_line_0.set_xdata(val[1])
        self.fig.canvas.draw_idle()
    
    def update_main_1(self, val):
        pass 
        
        











