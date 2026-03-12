import sys
import pandas as pd
import numpy as np  

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# from timeseriesplot import *
from timeseriesplot_v2 import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.resize(1920, 1080)
        self.setWindowTitle('Interactive Time Series Plot')
        self.dataframe = self.create_dataframe()
        
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        self.plot = TimeSeriesPlot(parent=self, dataframe=self.dataframe)
        self.layout.addWidget(self.plot)
        
        self.clear_button = QPushButton(text='Clear', parent=self)
        self.layout.addWidget(self.clear_button)
        self.clear_button.clicked.connect(self.clear_plot)
        
        # 创建和设置滑块
        self.setup_sliders()
        
        # * 这是给Plot对象整体的toolbar, 还需要在 Timeseriesplot 类中给每个子图各添加一个
        toolbar = NavigationToolbar(self.plot, self) 
        self.layout.addWidget(toolbar)
        
    def setup_sliders(self):
        # 创建并配置滑块
        self.start_slider_0 = QSlider(Qt.Horizontal)
        self.end_slider_0 = QSlider(Qt.Horizontal)
        self.start_slider_1 = QSlider(Qt.Horizontal)
        self.end_slider_1 = QSlider(Qt.Horizontal)

        max_index = len(self.dataframe['Time']) - 1
        for slider in [self.start_slider_0, self.end_slider_0, self.start_slider_1, self.end_slider_1]:
            slider.setMinimum(0)
            slider.setMaximum(max_index)
            self.layout.addWidget(slider)

        # 连接滑块信号到处理函数
        self.start_slider_0.valueChanged.connect(lambda value: self.plot.update_interval(0, 'start', value))
        self.end_slider_0.valueChanged.connect(lambda value: self.plot.update_interval(0, 'end', value))
        self.start_slider_1.valueChanged.connect(lambda value: self.plot.update_interval(1, 'start', value))
        self.end_slider_1.valueChanged.connect(lambda value: self.plot.update_interval(1, 'end', value))
        
        
        
        
        
    def create_dataframe(self):
        time_start = 0
        time_list = [time_start + i for i in range (1000)]
        np.random.seed(0)
        signal = np.sin(np.linspace(0, 20, 1000) + np.random.normal(0, 5, 1000))
        df = pd.DataFrame({'Time': time_list, 'Signal': signal})
        print(df.head())
        return df
        
    
    def clear_plot(self):
        pass
    
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
        
if __name__ == '__main__':
    main()




