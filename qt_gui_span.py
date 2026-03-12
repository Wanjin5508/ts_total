import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import datetime
from matplotlib.widgets import SpanSelector

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, dataframe=None):
        self.fig = Figure(figsize=(10, 8))
        super(TimeSeriesPlot, self).__init__(self.fig)
        self.ax_main = self.fig.add_subplot(211)
        self.ax_zoom = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)
        
        self.dataframe = dataframe
        self.plot_main()

        # Initialize SpanSelector without 'span_stays' parameter
        self.span = SpanSelector(
            self.ax_main,
            onselect=self.on_select,
            direction='horizontal',
            useblit=True,
            minspan=0.01,
            props=dict(alpha=0.5, facecolor='red'),
            interactive=True
        )

    def plot_main(self):
        self.ax_main.clear()
        self.ax_main.plot(self.dataframe['Time'], self.dataframe['Signal'], label='Signal Strength')
        self.ax_main.set_title('Time Series Signal Strength')
        self.ax_main.set_xlabel('Time')
        self.ax_main.set_ylabel('Signal Strength')
        self.ax_main.legend()
        self.ax_main.grid(True)

        # Improve date formatting if Time is datetime
        if isinstance(self.dataframe['Time'].iloc[0], (datetime.datetime, pd.Timestamp)):
            self.ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            self.fig.autofmt_xdate()

        self.draw()

    def plot_zoom(self, start_time, end_time):
        self.ax_zoom.clear()
        
        # 创建掩码，选择时间范围内的数据
        mask = (self.dataframe['Time'] >= start_time) & (self.dataframe['Time'] <= end_time)
        zoom_data = self.dataframe.loc[mask]
        
        print(f"Plot Zoom: Start Time = {start_time}, End Time = {end_time}")
        print(f"Number of data points in zoom range: {len(zoom_data)}")
        
        if not zoom_data.empty:
            self.ax_zoom.plot(zoom_data['Time'], zoom_data['Signal'], color='orange', label='Zoomed Signal')
            self.ax_zoom.set_title('Zoomed Signal Strength')
            self.ax_zoom.set_xlabel('Time')
            self.ax_zoom.set_ylabel('Signal Strength')
            self.ax_zoom.legend()
            self.ax_zoom.grid(True)

            if isinstance(self.dataframe['Time'].iloc[0], (datetime.datetime, pd.Timestamp)):
                self.ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                self.fig.autofmt_xdate()
        else:
            self.ax_zoom.text(0.5, 0.5, 'No data in selected range', 
                              horizontalalignment='center', verticalalignment='center')
            print("No data in the selected range.")

        self.draw()

    def on_select(self, xmin, xmax):
        """
        回调函数，当用户选择一个范围时触发。
        """
        # 将xmin和xmax从 Matplotlib 的日期数值转换为 Pandas Timestamp
        try:
            start_datetime = pd.to_datetime(mdates.num2date(xmin)).replace(tzinfo=None)
            end_datetime = pd.to_datetime(mdates.num2date(xmax)).replace(tzinfo=None)
            print(f"Selected range: {start_datetime} to {end_datetime}")
        except Exception as e:
            print(f"Error converting selection to datetime: {e}")
            return
        
        # 调用 plot_zoom 方法
        self.plot_zoom(start_datetime, end_datetime)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Time Series Plot with SpanSelector")

        # 创建示例 DataFrame
        self.dataframe = self.create_sample_data()

        # 设置主窗口的部件和布局
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # 初始化图表
        self.plot = TimeSeriesPlot(parent=self, dataframe=self.dataframe)
        self.layout.addWidget(self.plot)

        # 添加重置按钮
        self.reset_button = QPushButton("Reset Zoom")
        self.reset_button.clicked.connect(self.reset_zoom)
        self.layout.addWidget(self.reset_button)

    def create_sample_data(self):
        # 创建时间范围
        time_start = datetime.datetime.now()
        time_list = [time_start + datetime.timedelta(seconds=i) for i in range(1000)]
        
        # 生成带有噪声的随机信号数据
        np.random.seed(0)
        signal = np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.5, 1000)
        
        # 创建 DataFrame，并确保 Time 列为 tz-naive
        df = pd.DataFrame({'Time': pd.to_datetime(time_list).tz_localize(None), 'Signal': signal})
        
        # 打印 DataFrame 信息用于调试
        print("DataFrame created:")
        print(df.head())
        print(f"DataFrame 'Time' dtype: {df['Time'].dtype}")
        
        return df

    def reset_zoom(self):
        """
        重置放大图，显示完整数据。
        """
        print("Reset Zoom button clicked.")
        self.plot.plot_zoom(self.dataframe['Time'].min(), self.dataframe['Time'].max())

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
