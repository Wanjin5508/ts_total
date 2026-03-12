from PyQt5 import QtCore
import pyqtgraph as pg
from PyQt5.QtCore import Qt


class CommentLinearRegionItem(pg.LinearRegionItem):

    clicked = QtCore.pyqtSignal(object)
    # right_clicked_to_reset_exported = QtCore.pyqtSignal(object)
    ctrl_pressed = False

    def __init__(self, *args, recid=0, comment='new', is_preselected=False, is_exported=False, color_list=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.is_selected = False
        if len(color_list) != 0:
            color = color_list
        # else:
        #     color = [(255, 255, 0, 64), (255, 128, 0, 64)]
        #     self.default_brush = pg.mkBrush(255, 255, 0, 64)  # yellow
        #     self.selected_brush = pg.mkBrush(255, 128, 0, 64)  # orange
        self.default_brush = pg.mkBrush(color[0])  # by default, the selected intervals' color is from SELECTED_COLORS
        self.selected_brush = pg.mkBrush(color[-1])
        self.setBrush(self.default_brush)
        self.setHoverBrush(self.default_brush)  # deactivate hover effect
        self.setMovable(False)

        self.recid = recid
        self.signal_array = None
        self.comment = comment
        self.origin_start = 0.0
        self.origin_end = 0.0

        self.is_preselected = is_preselected
        self.is_exported = is_exported

        self.is_visible = True

    def set_comment(self, comment):
        self.comment = comment

    def get_comment(self):
        return self.comment

    def set_selected(self, selected):
        self.is_selected = selected
        self.setBrush(self.selected_brush if selected else self.default_brush)
        self.setHoverBrush(self.selected_brush if selected else self.default_brush)
        self.update()

    def on_right_click_unpick(self):
        self.is_selected = False
        self.setBrush(self.default_brush)
        self.setHoverBrush(self.default_brush)
        self.update()

    def set_exported(self, exported=True):
        self.is_exported = exported
        self.setMovable(False)
        self.setBrush(190, 190, 190, 60)  # grey
        self.update()

    def set_visible(self, is_visible):
        self.is_visible = is_visible
        self.setVisible(self.is_visible)

    def mouseClickEvent(self, event):
        # global current_selected_item

        if event.button() == 1 and not self.is_exported:
            # self.set_selected(not self.is_selected)
            self.clicked.emit(self)  # if this obj is clicked, then send it as a signal
            event.accept()
        elif event.button() == 2:
            print('right click to reset exported status!')
            if self.is_exported:
                self.clicked.emit(self)
                # self.redo_annotate_rect()
            event.ignore()
            return
        else:
            super().mouseClickEvent(event)

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key_Control:
    #         self.ctrl_pressed = True
    #     super().keyPressEvent(event)

    # def keyReleaseEvent(self, event):
    #     if event.key() == Qt.Key_Control:
    #         self.ctrl_pressed = False
    #     super().keyReleaseEvent(event)

    def mouseDragEvent(self, event):
        if not self.ctrl_pressed:
            self.setMovable(False)
            event.ignore()
            # event.accept()
        else:
            self.setMovable(True and not self.is_exported)
            if self.is_exported:
                self.setHoverBrush(190, 190, 190, 60)
            super().mouseDragEvent(event)

    def set_ctrl_pressed(self, state):
        self.ctrl_pressed = state
        if self.ctrl_pressed and not self.is_exported:
            self.setMovable(True)
        else:
            self.setMovable(False)

    def redo_annotate_rect(self):
        # use right key to reset exported grey rect into default color
        if not self.is_exported:
            return
        self.is_exported = False
        self.on_right_click_unpick()

    # def __eq__(self, other):
    #     if isinstance(other, CommentLinearRegionItem):
    #         return self.getRegion() == other.getRegion()

    # def __hash__(self):
    #     return hash(self.getRegion())

    def __repr__(self):
        return f"CommentLinearRegionItem(comment: {self.comment})"
