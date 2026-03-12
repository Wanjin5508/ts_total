import sys
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

class InteractivePlot(QWidget):
    def __init__(self, parent=None):
        super(InteractivePlot, self).__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 创建上方的图表
        self.fig_main = Figure(figsize=(5, 4))
        self.canvas_main = FigureCanvas(self.fig_main)
        self.ax_main = self.fig_main.add_subplot(111)
        layout.addWidget(self.canvas_main)
        
        # 创建下方的放大图
        self.fig_zoom = Figure(figsize=(5, 2))
        self.canvas_zoom = FigureCanvas(self.fig_zoom)
        self.ax_zoom = self.fig_zoom.add_subplot(111)
        layout.addWidget(self.canvas_zoom)
        
        # 生成示例数据
        self.x = np.linspace(0, 100, 1000)
        self.y = np.sin(self.x) + np.random.normal(0, 0.1, size=self.x.shape)
        
        # 绘制主图
        self.ax_main.plot(self.x, self.y, label='Signal Strength')
        self.ax_main.set_title('Signal Strength Time Series')
        self.ax_main.set_xlabel('Time')
        self.ax_main.set_ylabel('Signal Strength')
        self.ax_main.legend()
        
        # 存储选择的区域
        self.selected_regions = []
        self.current_selector = None
        
        # 激活矩形选择工具
        self.RS = RectangleSelector(self.ax_main, self.on_select,
                                     useblit=True,
                                    button=[1],  # 只响应左键
                                    minspanx=5, minspany=5,
                                    spancoords='data',
                                    interactive=True)
        
        # 连接点击事件以切换放大图
        self.canvas_main.mpl_connect('button_press_event', self.on_click)
        
        self.canvas_main.draw()
    
    def on_select(self, eclick, erelease):
        # 获取选择区域的坐标
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        # 绘制矩形
        rect = Rectangle((x_min, self.ax_main.get_ylim()[0]),
                         x_max - x_min, self.ax_main.get_ylim()[1] - self.ax_main.get_ylim()[0],
                         linewidth=1, edgecolor='red', facecolor='none', picker=True)
        self.ax_main.add_patch(rect)
        self.selected_regions.append({'rect': rect, 'x_min': x_min, 'x_max': x_max})
        self.canvas_main.draw()
    
    def on_click(self, event):
        # 检查点击是否在某个矩形内
        if event.inaxes != self.ax_main:
            return
        for region in self.selected_regions:
            rect = region['rect']
            contains = rect.contains(event)[0]
            if contains:
                self.update_zoom(region)
                break
    
    def update_zoom(self, region):
        self.ax_zoom.clear()
        x_min = region['x_min']
        x_max = region['x_max']
        mask = (self.x >= x_min) & (self.x <= x_max)
        self.ax_zoom.plot(self.x[mask], self.y[mask], label='Zoomed Signal')
        self.ax_zoom.set_title(f'Zoomed View: {x_min:.2f} - {x_max:.2f}')
        self.ax_zoom.set_xlabel('Time')
        self.ax_zoom.set_ylabel('Signal Strength')
        self.ax_zoom.legend()
        self.canvas_zoom.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = InteractivePlot()
    main.setWindowTitle("Interactive Signal Strength Viewer")
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
