import os, sys  
import pandas as pd  

from typing import Dict   
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt 

import pyqtgraph as pg

pg.setConfigOption('background', 'w')  # 白色背景
pg.setConfigOption('foreground', 'k')  # 黑色字体

# 字符串横坐标控件
class RotateAxisItem(pg.AxisItem): # 该类继承自 pg.AxisItem，用于创建自定义坐标轴
    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        """
        drawPicture 是 pyqtgraph.AxisItem 类中的绘图方法。重写此方法允许我们自定义坐标轴的绘制逻辑。--> 尽管官方技术文档中没提到, 但是可以用command查看源代码
        p: QPainter 对象，用于绘制图形。
        axisSpec: 包含轴线的绘制信息。包含一个画笔(pen)和两个端点坐标 p1, p2。
        tickSpecs: 包含刻度线的信息的列表, 每个元素包含一个画笔(pen)、两个端点坐标 p1, p2。
        textSpecs: 包含标签文本的信息。
        """
        
        # 渲染设置
        p.setRenderHint(p.Antialiasing, False) # 关闭抗锯齿, 提升绘图效率
        p.setRenderHint(p.Textaliasing, True) # 打开文本抗锯齿, 确保文本平滑显示
        
        # 绘制坐标轴线, 沿着坐标轴绘制长直线
        pen, p1, p2 = axisSpec
        p.setPen(pen)  # 使用指定的画笔
        p.drawLine(p1, p2)  # 绘制坐标轴线  
        p.translate(0.5, 0)  # 微调坐标, 消除可能的像素模糊
        
        # 绘制刻度线
        for pen, p1, p2 in tickSpecs: # 遍历 tickspecs 列表, 并用指定的画笔绘制每个刻度线
            p.setPen(pen)
            p.drawLine(p1, p2)
            
        # 绘制标签文本
        p.setPen(self.pen())  # 设置画笔为当前对象的画笔, 以绘制标签文本
        for rect, flags, text in textSpecs:   # 遍历 textSpecs，其中每个元素包含文本的 rect（位置矩形）、flags（对齐标志）、和 text（标签内容）。
            p.save() # 保存当前的绘图状态，以便稍后恢复
            p.translate(rect.x(), rect.y())  # 将绘图原点平移至标签位置。
            p.rotate(-30)  # 旋转标签文本 30°，实现倾斜效果
            p.drawText()
            p.drawText(int(-rect.width()), int(rect.height()), int(rect.width()), int(rect.height()), flags, text)
            p.restore()

class GraphWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_data()
        self.init_ui()
        
        
    def init_data(self):
        pass 
    
    def init_ui(self):
        pass 
    
    






