import sys
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import datetime

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, dataframe=None):
        
        self.fig = Figure(figsize=(10, 6))
        
        super(TimeSeriesPlot, self).__init__(self.fig)
        self.ax_main = self.fig.add_subplot(211)
        self.ax_zoom = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)
        
        self.dataframe = dataframe
        self.plot_main()

        # Variables to store mouse press and release positions
        self.press = None
        self.release = None

        # Connect the mouse events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)

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
        
        # Convert float date numbers to datetime and then to pandas Timestamp
        # Remove timezone information to make them tz-naive
        start_datetime = pd.to_datetime(mdates.num2date(start_time)).replace(tzinfo=None)
        end_datetime = pd.to_datetime(mdates.num2date(end_time)).replace(tzinfo=None)
        
        # Debug: Print the types to ensure compatibility
        # print(f"start_datetime: {start_datetime}, type: {type(start_datetime)}")
        # print(f"end_datetime: {end_datetime}, type: {type(end_datetime)}")
        
        mask = (self.dataframe['Time'] >= start_datetime) & (self.dataframe['Time'] <= end_datetime)
        zoom_data = self.dataframe.loc[mask]
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

        self.draw()

    def on_press(self, event):
        if event.inaxes != self.ax_main:
            return
        self.press = event.xdata
        # Optionally, add a vertical line or rectangle to show selection
        # For simplicity, not added here

    def on_release(self, event):
        if self.press is None or event.inaxes != self.ax_main:
            return
        self.release = event.xdata
        start_time = min(self.press, self.release)
        end_time = max(self.press, self.release)
        self.plot_zoom(start_time, end_time)
        self.press = None
        self.release = None

##########################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Time Series Plot")

        # Sample DataFrame creation
        self.dataframe = self.create_sample_data()

        # Set up the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize the plot
        self.plot = TimeSeriesPlot(parent=self, dataframe=self.dataframe)
        self.layout.addWidget(self.plot)

    def create_sample_data(self):
        # Create a time range
        time_start = datetime.datetime.now()
        time_list = [time_start + datetime.timedelta(seconds=i) for i in range(1000)]
        
        # Generate random signal data with some noise
        np.random.seed(0)
        signal = np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.5, 1000)
        
        # Create DataFrame with Time as pandas Timestamp (tz-naive)
        df = pd.DataFrame({'Time': pd.to_datetime(time_list).tz_localize(None), 'Signal': signal})
        return df

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
