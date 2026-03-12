import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import datetime
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

class DraggableRectangle:
    """
    Class to make Rectangle patches draggable and resizable.
    """
    def __init__(self, rect, on_select_callback):
        self.rect = rect
        self.press = None
        self.background = None
        self.on_select_callback = on_select_callback
        self.cid_press = rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.selected = False

    def on_press(self, event):
        if event.inaxes != self.rect.axes:
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        self.press = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.rect.axes:
            return
        x0, y0 = self.press
        dx = event.xdata - x0
        # Update rectangle width
        new_width = self.rect.get_width() + dx
        if new_width < 0:
            new_width = 0
        self.rect.set_width(new_width)
        self.press = (event.xdata, event.ydata)
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        if self.press is None:
            return
        self.press = None
        self.on_select_callback()

    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cid_press)
        self.rect.figure.canvas.mpl_disconnect(self.cid_release)
        self.rect.figure.canvas.mpl_disconnect(self.cid_motion)

class TimeSeriesPlot(FigureCanvas):
    def __init__(self, parent=None, dataframe=None):
        self.fig = Figure(figsize=(10, 8))
        super(TimeSeriesPlot, self).__init__(self.fig)
        self.ax_main = self.fig.add_subplot(211)
        self.ax_zoom = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.dataframe = dataframe
        self.plot_main()

        # List to store selected regions
        self.selected_regions = []

        # Initialize RectangleSelector
        self.RS = RectangleSelector(
            self.ax_main, self.on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='data',
            interactive=True
        )

        # Connect pick event for selecting regions
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

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

        mask = (self.dataframe['Time'] >= start_datetime) & (self.dataframe['Time'] <= end_datetime)
        zoom_data = self.dataframe.loc[mask]
        if not zoom_data.empty:
            self.ax_zoom.plot(zoom_data['Time'], zoom_data['Signal'], color='orange', label='Zoomed Signal')
            self.ax_zoom.set_title(f'Zoomed Signal Strength: {start_datetime} - {end_datetime}')
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

    def on_select(self, eclick, erelease):
        """
        Callback for RectangleSelector when a region is selected.
        """
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        # Add a Rectangle patch to the main axis
        rect_patch = Rectangle(
            (x_min, self.ax_main.get_ylim()[0]),
            x_max - x_min,
            self.ax_main.get_ylim()[1] - self.ax_main.get_ylim()[0],
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.3, picker=True
        )
        self.ax_main.add_patch(rect_patch)
        self.fig.canvas.draw()

        # Store the region
        region = {
            'x_min': x_min,
            'x_max': x_max,
            'patch': rect_patch,
            'draggable': DraggableRectangle(rect_patch, self.update_zoom)
        }
        self.selected_regions.append(region)

    def on_pick(self, event):
        """
        Callback for pick event when a region is clicked.
        """
        artist = event.artist
        for region in self.selected_regions:
            if region['patch'] == artist:
                self.highlight_region(region)
                self.plot_zoom(region['x_min'], region['x_max'])
                break

    def highlight_region(self, selected_region):
        """
        Highlight the selected region and unhighlight others.
        """
        for region in self.selected_regions:
            if region == selected_region:
                region['patch'].set_edgecolor('blue')
                region['patch'].set_alpha(0.5)
            else:
                region['patch'].set_edgecolor('red')
                region['patch'].set_alpha(0.3)
        self.fig.canvas.draw()

    def update_zoom(self):
        """
        Update the zoom plot based on the currently highlighted region.
        """
        for region in self.selected_regions:
            if region['patch'].get_edgecolor() == (0.0, 0.0, 1.0, 1.0):  # Blue edge
                self.plot_zoom(region['x_min'], region['x_max'])
                break

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
