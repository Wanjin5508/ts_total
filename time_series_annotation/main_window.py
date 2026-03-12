import sys
import os
import cProfile, pstats
import copy
import re

from PyQt5.QtGui import QKeyEvent
import pandas as pd
import numpy as np 
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QSplitter, QRadioButton,\
    QPushButton, QCheckBox, QSpinBox, QComboBox, QLineEdit, QLabel, QProgressBar, QButtonGroup, QAction, QFileDialog, QMessageBox,\
        QFormLayout, QWidget

import polars as pl
from timeseries_plot_qt import TimeSeriesPlot
from dataset_async import Dataset
from comment_rectangle import CommentLinearRegionItem
from color_scheme import *

StyleSheet = """
#ProgressBar {
    text-align: center;
    border-radius: 6px;
}
"""

class MainWindow(QMainWindow):
    # check_box_state = [True, True]
    # show_peaks_checkbox_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.menubar = self.menuBar()
        load_file_button_action = QAction("Load .npy or .npz Files", self)
        load_file_button_action.triggered.connect(self.on_load_click)
        # load_file_button_action.triggered.connect(select_files)
        # self.save = QAction("Save", self)
        self.file = self.menubar.addMenu("&File")
        self.file.addAction(load_file_button_action)
        # self.file.addAction(self.save)

        # to avoid reloading plot if we load a new file
        self.chunk_idx_before_reload = 0
        
        self.user_chunk_size = 1250000
        self.user_sample_rate = 1

        self.inner_init()

    def inner_init(self):
        self.setFocusPolicy(Qt.ClickFocus)
        # self.resize(1920, 1080)
        self.setWindowTitle("Interactive Time Series Annotation")

        # setup the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        splitter = QSplitter(Qt.Vertical)
        
        # initiallze dataset and interactive plot
        self.dataset = Dataset(chunk_size=self.user_chunk_size, sample_rate=self.user_sample_rate)
        # self.dataframe = self.dataset.get_sample() # use default settings to show the first chunk
        self.dataframe = self.dataset.create_sample_data(chunk_idx=self.chunk_idx_before_reload)

        self.plot = TimeSeriesPlot(parent=self, dataframe=self.dataframe)
        self.plot.showGrid(x=True, y=True)
        self.plot.setLimits(xMin=-0.2 * self.dataset.chunk_size,
                            xMax=1.2 * self.dataset.chunk_size,
                            yMin=1.5 * min([df['Signal'].min() for df in self.dataframe]),
                            yMax=1.5 * max([df['Signal'].max() for df in self.dataframe]))

        self.plot.line_region_item_list_update.connect(self.update_len_lr_item) # 信号.connect(槽函数)
        self.plot.get_offset_signal.connect(self.get_offset_signal_from_plot)
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.plot)
        # self.layout.addLayout(plot_layout)

        outer_layout = QHBoxLayout()
        # outer_layout.addStretch()

        groupbox_pre = QGroupBox("Interval Selection")
        groupbox_0 = QGroupBox("Plot and Calculation")
        groupbox_1 = QGroupBox("Import and Export")

        vlayout_preselection = QVBoxLayout()
        vlayout0 = QVBoxLayout()
        vlayout1 = QVBoxLayout()

        # * 1st column of widgets
        hlayout_recid_dist_height = QHBoxLayout()
        self.pre_selection_indicator_list = [False for i in range(len(Dataset.file_path))] # ! by default false, can only be converted in pick_all_preselected

        self.auto_pre_recid_combobox = QComboBox(self)
        self.auto_pre_recid_combobox.addItems([f"recid_" + str(i) for i in range(len(Dataset.file_path))])
        hlayout_recid_dist_height.addWidget(self.auto_pre_recid_combobox)
        self.auto_pre_recid_combobox.activated.connect(self.on_auto_pre_recid)

        self.step = 1
        hlayout_label_slider = QHBoxLayout()
        self.label_seg_slider = QLabel("Distance (x) = ")
        self.distance = [100 for i in range(len(Dataset.file_path))]
        hlayout_label_slider.addWidget(self.label_seg_slider)

        self.distance_line_edit = QLineEdit(self)
        self.distance_line_edit.setText(str(self.distance[0]))
        hlayout_label_slider.addWidget(self.distance_line_edit)
        self.distance_line_edit.returnPressed.connect(self.on_distance_given)
        hlayout_recid_dist_height.addLayout(hlayout_label_slider)

        # height
        hlayout_height_linedit = QHBoxLayout()
        self.label_height = QLabel("Height (y) = ")
        hlayout_height_linedit.addWidget(self.label_height)

        self.height = [100 for _ in range(len(Dataset.file_path))]
        self.height_line_edit = QLineEdit(self)
        # self.height_line_edit.setPlaceholderText(str(self.height[0]))
        self.height_line_edit.setText(str(self.height[0]))
        hlayout_height_linedit.addWidget(self.height_line_edit)
        self.height_line_edit.returnPressed.connect(self.on_height_given)
        vlayout_preselection.addLayout(hlayout_height_linedit)

        # window size
        # hlayout_detect_sustained_spikes = QHBoxLayout()
        self.label_window_size = QLabel("Window size =")
        hlayout_recid_dist_height.addWidget(self.label_window_size)
        self.window_size_list = [400 for _ in range(len(Dataset.file_path))]
        self.window_size_line_edit = QLineEdit(self)
        self.window_size_line_edit.setText(str(self.window_size_list[0]))
        hlayout_label_slider.addWidget(self.window_size_line_edit)
        self.window_size_line_edit.returnPressed.connect(self.on_window_size_given)
        
        # self.label_threshold = QLabel("Threshold = ")
        # hlayout_label_slider.addWidget(self.label_threshold)
        self.sustained_thresholds_list = [100 for _ in range(len(Dataset.file_path))]
        # vlayout_preselection.addLayout(hlayout_detect_sustained_spikes)
        

        # envelope
        hlayout_origin_envelope_selec = QHBoxLayout()
        self.label_line_selection = QLabel("Select reference line list:")
        self.reference_line_combobox = QComboBox(self)
        self.reference_line_combobox.addItems(['Max Envelope (grey)', 'SSM Envelope (red)', 'RMS Envelope (green)'])
        hlayout_origin_envelope_selec.addWidget(self.label_line_selection)
        hlayout_origin_envelope_selec.addWidget(self.reference_line_combobox)
        # self.reference_line_combobox.activated.connect(self.on_envelope_select)
        vlayout_preselection.addLayout(hlayout_origin_envelope_selec)

        # preselection buttons
        hlayout_autoselection_and_pick = QHBoxLayout()
        self.preselection_button = QPushButton(text="Automated interval pre-selection")
        hlayout_autoselection_and_pick.addWidget(self.preselection_button)
        self.preselection_button.clicked.connect(self.preselection)

        self.all_preselected_button = QPushButton(text="Pick all intervals")
        hlayout_autoselection_and_pick.addWidget(self.all_preselected_button)
        self.all_preselected_button.clicked.connect(self.pick_all_preselected)

        self.undo_pick_all_button = QPushButton(text="Undo pick")
        hlayout_autoselection_and_pick.addWidget(self.undo_pick_all_button)
        self.undo_pick_all_button.clicked.connect(self.undo_pick_all)

        vlayout_preselection.addLayout(hlayout_autoselection_and_pick)
                # lower and upper limits
        hlayout_lower_upper_limit = QHBoxLayout()
        # self.label_reset_range_left = QLabel(text="Reset lower limit of interval (s)")
        # vlayout_preselection.addWidget(self.label_reset_range_left)
        self.extend_lower_limit_button = QPushButton(text="<<<")
        self.extend_lower_limit_lineedit = QLineEdit(self)
        self.extend_lower_limit_lineedit.setText(str(100))
        
        self.extend_upper_limit_lineedit = QLineEdit(self)
        self.extend_upper_limit_lineedit.setText(str(100))
        self.extend_upper_limit_button = QPushButton(text=">>>")
        
        hlayout_lower_upper_limit.addWidget(self.extend_lower_limit_button)
        hlayout_lower_upper_limit.addWidget(self.extend_lower_limit_lineedit)
        hlayout_lower_upper_limit.addWidget(self.extend_upper_limit_lineedit)
        hlayout_lower_upper_limit.addWidget(self.extend_upper_limit_button)
        
        vlayout_preselection.addLayout(hlayout_lower_upper_limit)

        self.extend_lower_limit_button.clicked.connect(self.on_extend_lower_gui)
        self.extend_upper_limit_button.clicked.connect(self.on_extend_upper_gui)
        
        vlayout_preselection.addStretch(1)

        # * 2nd column of widgets
        # clear & patch delete
        hlayout_clear_delete = QHBoxLayout()
        self.clear_button = QPushButton(text="Clear", parent=self)
        hlayout_clear_delete.addWidget(self.clear_button)
        self.clear_button.clicked.connect(self.clear_plot)

        self.delete_patch_button = QPushButton(text="Delete Patch", parent=self)
        hlayout_clear_delete.addWidget(self.delete_patch_button)
        self.delete_patch_button.clicked.connect(self.delete_patch)
        vlayout0.addLayout(hlayout_clear_delete)

        # chunk size
        hlayout_set_chunk_size = QHBoxLayout()
        self.set_chunk_size_label = QLabel(text="Chunk size = ")
        hlayout_set_chunk_size.addWidget(self.set_chunk_size_label)

        self.set_chunk_size_lineedit = QLineEdit(self)
        self.set_chunk_size_lineedit.setPlaceholderText(str(self.dataset.chunk_size))
        hlayout_set_chunk_size.addWidget(self.set_chunk_size_lineedit)
        self.set_chunk_size_lineedit.returnPressed.connect(self.on_chunk_size_given)

        self.set_sample_rate_label = QLabel(text="Down sampling = ")
        hlayout_set_chunk_size.addWidget(self.set_sample_rate_label)
        
        self.set_sample_rate_lineedit = QLineEdit(self)
        self.set_sample_rate_lineedit.setPlaceholderText(str(self.dataset.sample_rate))
        hlayout_set_chunk_size.addWidget(self.set_sample_rate_lineedit)
        self.set_sample_rate_lineedit.returnPressed.connect(self.on_sample_rate_given)

        self.update_chunk_sample_button = QPushButton(text="Update")
        hlayout_set_chunk_size.addWidget(self.update_chunk_sample_button)
        self.update_chunk_sample_button.pressed.connect(self.update_chunk_sample)
        vlayout0.addLayout(hlayout_set_chunk_size)

        # move line
        hlayout_move_line = QHBoxLayout()
        self.move_line_label = QLabel(text=f"recid_{self.auto_pre_recid_combobox.currentIndex()}")
        self.radio_btn_left = QRadioButton("left", self)
        self.radio_btn_left.setChecked(True)
        self.radio_btn_right = QRadioButton("right", self)
        self.radio_btn_group = QButtonGroup(self)
        self.radio_btn_group.addButton(self.radio_btn_left, 0)
        self.radio_btn_group.addButton(self.radio_btn_right, 1)
        
        self.nr_samples_move_lineedit = QLineEdit(self)
        self.nr_samples_move_lineedit.setText(str(100))
        self.line_move_btn = QPushButton(text="Move", parent=self)
        hlayout_move_line.addWidget(self.move_line_label)
        hlayout_move_line.addWidget(self.radio_btn_left)
        hlayout_move_line.addWidget(self.radio_btn_right)
        hlayout_move_line.addWidget(self.nr_samples_move_lineedit)
        hlayout_move_line.addWidget(self.line_move_btn)
        vlayout0.addLayout(hlayout_move_line)
        self.radio_btn_group.buttonClicked.connect(self.rbclicked)
        self.line_move_btn.pressed.connect(self.move_line)

        # offset label
        self.current_offset_label = QLabel(text="current offset:\t0")
        self.current_offset_label.setFixedWidth(300)
        hlayout_move_line.addWidget(self.current_offset_label)

        # reset offset
        self.reset_x_offset_btn = QPushButton(text="Reset x offset")
        self.reset_x_offset_btn.pressed.connect(self.reset_x_offset)
        hlayout_move_line.addWidget(self.reset_x_offset_btn)

        # traveling waves - cable length calc
        hlayout_cable_length_traveling_waves = QHBoxLayout()
        grid_layout_traveling_waves = QGridLayout()
        self.cable_length_label = QLabel(text="Cable length (m) = ")
        self.cable_length_lineedit = QLineEdit(self)
        self.cable_length_label.setFixedWidth(250)
        self.cable_length_lineedit.setText(str(3160))
        
        self.wave_speed_label = QLabel(text="Wave speed (1e9 s/m) = ")
        self.wave_speed_lineedit = QLineEdit(self)
        self.wave_speed_lineedit.setText(str(6))
        self.wave_speed_label.setFixedWidth(250)
        # self.calc_distance_btn = QPushButton(text="Get Distance")
        self.show_distance_label = QLabel(text="Distance =  0 m")

        grid_layout_traveling_waves.addWidget(self.cable_length_label, 0, 0)
        grid_layout_traveling_waves.addWidget(self.cable_length_lineedit, 0, 1)
        grid_layout_traveling_waves.addWidget(self.wave_speed_label, 1, 0)
        grid_layout_traveling_waves.addWidget(self.wave_speed_lineedit, 1, 1)

        hlayout_cable_length_traveling_waves.addLayout(grid_layout_traveling_waves, 5)
        hlayout_on_right_grid = QHBoxLayout()
        hlayout_on_right_grid.addWidget(self.show_distance_label)
        hlayout_cable_length_traveling_waves.addLayout(hlayout_on_right_grid, 4)
        vlayout0.addLayout(hlayout_cable_length_traveling_waves)

        # self.calc_distance_btn.pressed.connect(self.on_calc_distance_btn)
        vlayout0.addStretch()
        
        # *3rd column of widgets
        # --- 第 3 列控件：chunk 跳转与文件操作 ---
        # chunk 控制
        form_layout_and_button = QHBoxLayout()
        form_layout0 = QFormLayout()

        go_to_button = QPushButton(text="Go to >>>")
        go_to_button.clicked.connect(self.go_to_chunk)

        self.next_chunk_btn = QPushButton("Next chunk")
        self.next_chunk_btn.clicked.connect(self.next_chunk)

        self.spinbox = QSpinBox(parent=self)
        self.spinbox.setRange(0, self.dataset.chunk_cnt)
        self.spinbox.setValue(self.chunk_idx_before_reload)
        self.spinbox.setSingleStep(1)
        self.spinbox.resize(20, 10)
        self.spinbox.valueChanged.connect(self.value_changed)

        form_layout0.addRow(f"Select chunk index between 0 and {self.dataset.chunk_cnt}: ", self.spinbox)
        form_layout_and_button.addLayout(form_layout0)
        form_layout_and_button.addWidget(go_to_button)
        form_layout_and_button.addWidget(self.next_chunk_btn)
        vlayout1.addLayout(form_layout_and_button)

        # --- 文件选择与标注 ---
        hlayout = QHBoxLayout()
        self.combo = QComboBox(self)

        for path in Dataset.file_path:
            item_list = path.split("/")[-2:]
            item = "/".join(str(x) for x in item_list)
            item = item.split(".")[0]
            self.combo.addItem(item)

        self.combo.activated.connect(self.on_combo_activated)
        hlayout.addWidget(self.combo)

        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText("<Add a comment and press enter to store>")
        self.line_edit.returnPressed.connect(self.on_line_return_pressed)
        hlayout.addWidget(self.line_edit)

        # 多区间注释
        hlayout_multi_intervals_comment = QHBoxLayout()
        vlayout1.addLayout(hlayout)

        # --- 导入与导出 ---
        hlayout_import_export = QHBoxLayout()
        self.import_existing_parquet_button = QPushButton(text="Import existing .parquet")
        hlayout_import_export.addWidget(self.import_existing_parquet_button)
        self.import_existing_parquet_button.clicked.connect(self.import_parquet)

        self.store_selected_interval_button = QPushButton(text="Export as .parquet")
        hlayout_import_export.addWidget(self.store_selected_interval_button)
        self.store_selected_interval_button.clicked.connect(self.export_parquet)

        self.export_to_existing_parquet_button = QPushButton(text="Export to an existing file")
        hlayout_import_export.addWidget(self.export_to_existing_parquet_button)
        self.export_to_existing_parquet_button.clicked.connect(self.export_to_existing_parquet)
        vlayout1.addLayout(hlayout_import_export)

        # --- 工具栏：导航与状态显示 ---
        toolbar_layout = QHBoxLayout()
        self.number_of_preselected_interval = QLabel(
            text=f"NO. Selected intervals of recid_{self.auto_pre_recid_combobox.currentIndex()}: Pre = 0, Manual = 0\t"
        )
        toolbar_layout.addWidget(self.number_of_preselected_interval)

        self.show_peaks_checkbox = QCheckBox("Show Peaks and Envelope    ")
        toolbar_layout.addWidget(self.show_peaks_checkbox)
        self.show_peaks_checkbox.setChecked(False)
        self.show_peaks_checkbox.stateChanged.connect(self.on_show_peaks)

        # 多通道复选框
        self.checkbox_list = [QCheckBox("recid_" + str(i)+"  ") for i in range(len(Dataset.file_path))]
        for checkbox in self.checkbox_list:
            checkbox.setChecked(True)
            toolbar_layout.addWidget(checkbox)
            checkbox.stateChanged.connect(self.checkbox_state_changed)
        toolbar_layout.addStretch()

        # 偏移模式
        self.offset_mode_checkbox = QCheckBox("Offset mode")
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.offset_mode_checkbox)
        self.offset_mode_checkbox.setChecked(False)
        self.offset_mode_checkbox.stateChanged.connect(self.on_offset_mode_change)

        # 进度条
        self.pbar = QProgressBar(self, objectName="ProcessBar", textVisible=False)
        toolbar_layout.addWidget(self.pbar)

        # --- 布局汇总 ---
        vlayout1.addStretch()
        groupbox_pre.setLayout(vlayout_preselection)
        groupbox_0.setLayout(vlayout0)
        groupbox_1.setLayout(vlayout1)

        # outer_layout.addLayout(vlayout_preselection)
        # outer_layout.addLayout(vlayout0)
        # outer_layout.addLayout(vlayout1)
        outer_layout.addWidget(groupbox_pre)
        outer_layout.addWidget(groupbox_0)
        outer_layout.addWidget(groupbox_1)

        under_layout = QVBoxLayout()
        under_layout.addLayout(toolbar_layout, 1)
        under_layout.addLayout(outer_layout, 4)

        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)

        control_widget = QWidget()
        control_widget.setLayout(under_layout)
        # control_widget.setMinimumHeight(100)

        splitter.addWidget(plot_widget)
        splitter.addWidget(control_widget)
        splitter.setSizes([800, 200])

        self.layout.addWidget(splitter)
        self.setLayout(self.layout)
        self.showMaximized()

    # =========================================================
    #                动作函数区
    # =========================================================
    
    def update_label_seg(self, value):
        custom_value = value * self.step
        self.label_seg_slider.setText(f"Segment length: {custom_value}")
        self.distance = custom_value
    
    def update_len_lr_item(self, length):
        # slot
        recid = self.auto_pre_recid_combobox.currentIndex()
        self.number_of_preselected_interval.setText(f"No. Selected intervals of recid_{recid}: Pre = {len(set(self.plot.pre_intervals[recid]))}, Manual = {length}\t")
    
    def on_load_click(self, s):
        self.chunk_idx_before_reload = self.spinbox.value()
        # print("click", s)
        desktop_path = os.path.expanduser("~/Desktop")
        files, dialog = QFileDialog.getOpenFileNames(
            self, 
            "Select Files", 
            desktop_path, 
            "npy Files (*.npy);; npz Files (*.npz)"
        )
        if not files:
            return
        for file in files:
            if file in Dataset.file_path:
                continue
            if len(Dataset.file_path) >= 8:
                print("Maximum 8 files supported.")
                return
            Dataset.file_path.append(file)

        # Dataset.file_path = list(set(Dataset.file_path))
        
        Dataset.file_cnt = len(Dataset.file_path)
        # Dataset.file_path.sort()

        # 重新初始化 UI
        self.inner_init()

    def clear_plot(self):
        self.plot.update_plot()
        
        for i, line in enumerate(self.plot.plot_data_item_list):
            line.setVisible(self.checkbox_list[i].isChecked())
            self.plot.toggle_line(self.plot.peaks_list[i], self.show_peaks_checkbox.isChecked() and self.checkbox_list[i].isChecked())
            self.plot.toggle_line(self.plot.line_envelope_list[i], self.show_peaks_checkbox.isChecked() and self.checkbox_list[i].isChecked())
            self.plot.toggle_line(self.plot.line_ssm_envelope_list[i], self.show_peaks_checkbox.isChecked() and self.checkbox_list[i].isChecked())
            self.plot.toggle_line(self.plot.rms_envelope_list[i], self.show_peaks_checkbox.isChecked() and self.checkbox_list[i].isChecked())
            
            
        # * to reset the list of selected intervals in this chunk
        for item_list in self.plot.line_region_item_list+self.plot.pre_intervals:
            item_list = [self.plot.removeItem(item) for item in item_list]
            
        self.plot.line_region_item_list = [[] for _ in range(len(self.dataframe))]
        self.plot.line_region_item_list_update.emit(len(self.plot.line_region_item_list[self.plot.get_auto_pre_recid()]))

        self.plot.pre_intervals = copy.deepcopy(self.plot.line_region_item_list)
        self.number_of_preselected_interval.setText(f"NO. Selected intervals of recid_{self.auto_pre_recid_combobox.currentIndex()}: Pre = {len(set(self.plot.pre_intervals[self.auto_pre_recid_combobox.currentIndex()]))}, Manual = {len(set(self.plot.line_region_item_list[self.auto_pre_recid_combobox.currentIndex()]))}\t")
        self.plot.picked_lr_items.clear()


    def delete_patch(self):
        """删除选中矩形区域"""
        if self.plot.selected_interval is None and len(self.plot.picked_lr_items) == 0:
            QMessageBox.warning(self, "Warning", "No intervals selected!")
            return

        recid = self.auto_pre_recid_combobox.currentIndex()
        to_remove = self.plot.picked_lr_items
        
        for item in to_remove:
            if item.is_preselected:
                self.plot.pre_intervals[recid].remove(item)
            else:
                self.plot.line_region_item_list[recid].remove(item)
            self.plot.removeItem(item)
        self.plot.picked_lr_items.clear()
        self.number_of_preselected_interval.setText(f"NO. Selected intervals of recid_{recid}: Pre = {len(set(self.plot.pre_intervals[recid]))}, Manual = {len(set(self.plot.line_region_item_list[srecid]))}\t")

    def on_chunk_size_given(self):
        text = self.set_chunk_size_lineedit.text()
        if len(text) == 0:
            return
        try:
            text = text.replace(",", ".")
            cleaned_text = re.sub(r"[^0-9]", "", text)
            if cleaned_text:
                self.user_chunk_size = int(cleaned_text)
                self.plot.chunk_size = self.user_chunk_size
                self.set_chunk_size_lineedit.setText(str(self.user_chunk_size))
            else:
                raise ValueError("Invalid input!")
        except ValueError as e:
            QMessageBox.warning(self, "{e}")

    def on_sample_rate_given(self):
        text = self.set_sample_rate_lineedit.text().replace(",", "").strip()
        if not text:
            return
        try:
            value = int(text)
            self.user_sample_rate = value
            Dataset.rate = value
            self.set_sample_rate_lineedit.setText(str(value))
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid sample rate input!")

    def rbclicked(self):
        sender = self.sender()
        # if self.radio_btn_left.isChecked():
        #     print("Left button selected")
        # else:
        #     print("Right button selected")
        if sender == self.radio_btn_group:
            if self.radio_btn_group.checkedId() == 0:
                print('the left btn is checked')
            else:
                print('the right')

    def move_line(self):
        recid = self.auto_pre_recid_combobox.currentIndex()
        rb_id = self.radio_btn_group.checkedId()
        direction = -1 if rb_id == 0 else 1
        moving_range = int(self.nr_samples_move_lineedit.text())
        self.plot.shift_plot(recid, direction, moving_range)
        self.current_offset_label.setText(
            f"current offset:\t{self.plot.x_offset_list[recid]}"
        )

    def reset_x_offset(self):
        self.plot.x_offset_list = [0 for _ in range(len(self.dataframe))]
        self.plot.update_plot()
        self.nr_samples_move_lineedit.setText(str(100))
        self.current_offset_label.setText("current offset:\t{self.plot.x_offset_list[self.auto_pre_recid_combobox.currentIndex()]}")
        
        self.plot.removeItem(self.plot.offset_indicator_item)
        self.plot.removeItem(self.plot.offset_indicator_item_limit_line_0)
        self.plot.removeItem(self.plot.offset_indicator_item_limit_line_1)
        self.plot.offset_indicator_item = None
        self.plot.offset_indicator_item_limit_line_0 = None
        self.plot.offset_indicator_item_limit_line_1 = None
        

    def get_offset_signal_from_plot(self, offset: tuple):
        start_coord, end_coord = offset # start_coord -> recid0, end_coord -> 1
        self.delta_x_in_samples = int(start_coord - end_coord) 
        self.nr_samples_move_lineedit.setText(str(self.delta_x_in_samples))
        
        distance = self.on_calc_distance_btn()

    def on_calc_distance_btn(self):
        cable_length = self.parse_float_from_str(self.cable_length_lineedit.text())
        wave_speed = self.parse_float_from_str(self.wave_speed_lineedit.text())
        if cable_length and wave_speed:
            # delta_t = self.delta_x_in_samples / self.user_sample_rate   # in nano seconds
            delta_t = self.delta_x_in_samples * 16 * 1e-9   # nano seconds
            distance = self.calc_distance_with_formel(cable_length, delta_t, wave_speed)
            self.show_distance_label.setText(f"Distance = {round(distance, 3)} m")

            self.plot.left_plotId_sampleIdx_tuple = None
            self.plot.right_plotId_sampleIdx_tuple = None
            return distance
        
    def parse_float_from_str(self, text:str):
        text = text.replace(",", ".")
        try:
            return float(text)
        except ValueError:
            print(f'Error: parse_float_from_str cannot parse {text}')
            return None
        
    def calc_distance_with_formel(self, cable_length:float, delta_t:int, wave_speed:float) -> float:
        if wave_speed == 0:
            return
        wave_speed *= 1e-9   # s/m

        return (cable_length + delta_t * (1 / wave_speed)) * 0.5

    def update_chunk_sample(self):
        self.on_chunk_size_given()
        self.on_sample_rate_given()
        self.inner_init()
        # self.set_chunk_size_lineedit.setText(str(self.user_chunk_size))


    def on_key_up(self):
        """
        extend both the left and right boundries of Rect
        """
        # if self.plot.selected_interval is None:
        #     return
        if len(self.plot.picked_lr_items) == 0:
            return
        text = self.extend_lower_limit_lineedit.text()
        try:
            range = int(text)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Could not convert '{text}' to integer: {e}")
            return

        for item in self.plot.picked_lr_items:
            lower, upper = item.getRegion()
            # lower = self.on_extend_lower_gui()
            # upper = self.on_extend_upper_gui()
            lower -= range
            upper += range
            item.setRegion([lower, upper])

    def on_key_down(self):
        """
        """
        if len(self.plot.picked_lr_items) == 0:
            return
        text = self.extend_lower_limit_lineedit.text()
        try:
            range = int(text)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Could not convert '{text}' to integer: {e}")
            return

        # range = int(text)
        for item in self.plot.picked_lr_items:
            lower, upper = item.getRegion()
            # lower = self.on_extend_lower_gui()
            # upper = self.on_extend_upper_gui()
            lower += range
            upper -= range
            item.setRegion([lower, upper])
            
    def on_lower_extend(self, range=0):
        """left key"""
        for rect in self.plot.picked_lr_items:
            lower, upper = rect.getRegion()
            if range == 0:
                range = 200
            lower -= range
            rect.setRegion([lower, upper])
            
    def on_upper_extend(self, range=0):
        """right key"""
        for rect in self.plot.picked_lr_items:
            lower, upper = rect.getRegion()
            if range == 0:
                range = 200
            upper += range
            rect.setRegion([lower, upper])
            
    def on_extend_lower_gui(self):
        text = self.extend_lower_limit_lineedit.text()
        try:
            range = int(text)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Could not convert '{text}' to integer: {e}")
            return
        range = int(text)
        self.on_lower_extend(range)
        return range
    
    def on_extend_upper_gui(self):
        text = self.extend_upper_limit_lineedit.text()
        try:
            range = int(text)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Could not convert '{text}' to integer: {e}")
            return
        range = int(text)
        self.on_upper_extend(range)
        return range

    def keyPressEvent(self, event) -> None:
        # print(f"Pressed key: {event.key()}")
        if event.key() == Qt.Key_Delete:
            # use del key to delete selected interval
            self.delete_patch()
            event.accept()  # mark the event as handled
            return

        elif event.key() == Qt.Key_Up:
            self.on_key_up()
            event.accept()
            return
        elif event.key() == Qt.Key_Down:
            self.on_key_down()
            event.accept()
            return

        elif event.key() == Qt.Key_Left:
            self.on_lower_extend(range=int(self.extend_lower_limit_lineedit.text()))
            event.accept()
            return

        elif event.key() == Qt.Key_Right:
            self.on_upper_extend(range=int(self.extend_upper_limit_lineedit.text()))
            event.accept()
            return

        elif event.key() == Qt.Key_PageUp:
            self.next_chunk()
            return
        elif event.key() == Qt.Key_PageDown:
            self.last_chunk()
            return

        elif event.key() == Qt.Key_Control:
            CommentLinearRegionItem.ctrl_pressed = True
            # for rect in self.plot.line_region_item_list[self.auto_pre_recid_combobox.currentIndex()]:
            #     rect.set_ctrl_pressed(True)
            return

        else:
            super(MainWindow, self).keyPressEvent(event)
            return

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            CommentLinearRegionItem.ctrl_pressed = False
            # for rect in self.plot.pre_intervals[self.auto_pre_recid_combobox.currentIndex()]:
            #     rect.set_ctrl_pressed(False)
            return super().keyReleaseEvent(event)

    def on_combo_activated(self, index):
        print(f"Combo is changed into {index}")
        # use index to get original data from dataset.arraylist
        print(f"(self.combo.currentIndex(): {self.combo.currentIndex()})")
        print(self.combo.currentText())

    def on_auto_pre_recid(self, index):
        print(f"auto preselection recid = {index}")
        self.plot.set_auto_pre_recid(index)
        self.distance_line_edit.setText(str(self.distance[index]))
        self.height_line_edit.setText(str(self.height[index]))
        self.window_size_line_edit.setText(str(self.window_size_list[index]))
        self.move_line_label.setText(f"(recid: {index})")
        self.combo.setCurrentIndex(self.auto_pre_recid_combobox.currentIndex())
        self.undo_pick_all()

    def store_selected_interval(self):
        if self.plot.selected_interval is None and len(self.plot.picked_lr_items) == 0:
            QMessageBox.warning(self, "Note", "You have not selected any intervals!")
            return

        self.step = 0
        for rect in self.plot.picked_lr_items:
            start, end = rect.getRegion()
            start, end = int(start), int(end)
            recid = rect.recid
            offset = self.plot.x_offset_list[recid]

            # remove the offset
            start, end = start - offset, end - offset
            array = self.dataset.array_list[recid]
            array_origin, original_start, original_end = self.dataset.back2origin(array, start, end, self.spinbox.value())
            rect.signal_array = array_origin
            rect.origin_start = original_start
            rect.origin_end = original_end

            try:
                self.plot.line_region_item_list[recid].remove(rect)
            except ValueError as e:
                print("cannot remove from manually selected intervals list")

            try:
                self.plot.pre_intervals[recid].remove(rect)
            except ValueError as e:
                print("cannot remove from auto")

            self.plot.line_region_item_list_update.emit(len(self.plot.line_region_item_list[recid]))
            self.number_of_preselected_interval.setText(f"NO. Selected intervals of recid_{recid}: Pre = {len(set(self.plot.pre_intervals[recid]))}, Manuel = {len(set(self.plot.line_region_item_list[recid]))}\t")

            self.step += 1
            self.pbar.setValue(self.step)
            self.pbar.reset()

        # print(f'store_selected_interval: the picked interval in manual/auto list? ({rect in self.plot.line_region_item_list[self.auto_pre_recid_combobox.currentIndex()]}, rect in self.plot.pre_intervals[self.auto_pre_recid_combobox.currentIndex()]})')
        print(f'len picked: {len(self.plot.picked_lr_items)}')

    def import_parquet(self):
        file_path = self.load_existing_file_path()
        self.plot.int_to_export_rects_dict()

        imported_df = pl.read_parquet(file_path)
        self.pbar.setMinimum(0)
        self.pbar.setMaximum(len(imported_df))
        step = 0

        # iterate over the file, and re-build self.plot.to_export_rects_dict
        for row in imported_df.iter_rows(named=False):   # if True, then expensive
            """
            0: recid, int
            1: signal, [int]
            2: comments, str
            3: file_path, str
            4: range, [int, int]
            """
            recid = row[0]
            start_origin, end_origin = row[4]
            try:
                chunk_id, lower_idx_in_chunk, upper_idx_in_chunk = self.dataset.origin2chunk(start_origin, end_origin)
            except TypeError as e:
                print(f'lower and upper limits conflict {e}')
                continue

            try:
                index_in_combo = Dataset.file_path.index(row[3])
            except ValueError as e:
                print("file was not loaded!")
                continue

            rect = CommentLinearRegionItem([lower_idx_in_chunk, upper_idx_in_chunk], recid=index_in_combo, clipItem=self.plot.plot_data_item_list[index_in_combo], color_list=[SELECTED_COLORS[index_in_combo], PICKED_COLORS[index_in_combo]])
            rect.set_exported(True)
            rect.set_visible(self.checkbox_list[index_in_combo].isChecked())
            self.plot.to_export_rects_dict[index_in_combo][chunk_id].append(rect)

            step += 1
            self.pbar.setValue(step)
            self.go_to_chunk()
        self.pbar.reset()

    def export_parquet(self):
        chunk_id = self.spinbox.value()
        file_path_combo_current_text = self.combo.currentText()
        dir = file_path_combo_current_text.split("_")[0]
        target_name = dir + "_" + str(chunk_id) + "_parquet"
        self.dataset.output_df_export_to_parquet(target_name=target_name)
        # print(f'file name with chunk_id {chunk_id}')

    def export_to_existing_parquet(self):
        # default_dir = self.dataset.output_df_default_export_dir
        file = self.load_existing_file_path()
        # print(pl.read_parquet(file))
        self.dataset.output_df.export_to_parquet(file)

    def load_existing_file_path(self) -> str:
        default_dir = self.dataset.output_df.default_export_dir
        file, dialog = QFileDialog.getOpenFileName(self, 
                                                   "Select an existing parquet file", 
                                                   default_dir, 
                                                   "Parquet Files (*.parquet);;Text Files (*.txt)")
        return file

    # @timer
    def preselection(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # clear all the preselected intervals
        for rect_list in self.plot.pre_intervals:
            rect_list = [self.plot.removeItem(rect) for rect in rect_list if not rect.is_exported]
            rect_list = []

        self.on_height_given()
        self.on_distance_given()
        # self.on_threshold_given()
        self.on_window_size_given()
        recid = self.auto_pre_recid_combobox.currentIndex()
        distance = self.distance[recid] // self.user_sample_rate
        height = self.height[recid]
        window_size = self.window_size_list[recid]
        
        # threshold = self.sustained_thresholds_list[recid]
        
        # TODO now we have the id=3, which means tcnae anomaly score
        envelope_id = self.reference_line_combobox.currentIndex()

        # print(f'mainwindow preselection(): recid, distance, height, envelope_id = {recid, distance, height, envelope_id}')

        for idx in range(len(self.plot.peaks_list)):
            self.plot.removeItem(self.plot.line_envelope_list[idx])
            self.plot.removeItem(self.plot.line_ssm_envelope_list[idx])
            self.plot.removeItem(self.plot.line_rms_envelope_list[idx])
            self.plot.removeItem(self.plot.peaks_list[idx])

        self.plot.update()
        
        
        # ! executed by PlotThread
        self.plot.preselection(recid=recid, distance=distance, height=height, show_peaks=self.show_peaks_checkbox.isChecked(), window_size=window_size, threshold=height, envelope_id=envelope_id)
        print(f'length of the preselected interval-list = {len(self.plot.pre_intervals[recid])}')

        self.number_of_preselected_interval.setText(f"NO. Selected intervals of recid_{recid}: Pre = {len(set(self.plot.pre_intervals[recid]))}, Manuel = {len(self.plot.line_region_item_list[recid])}\t")
        QApplication.restoreOverrideCursor()

    def pick_all_preselected(self):
        self.plot.picked_lr_items.clear()
        idx = self.auto_pre_recid_combobox.currentIndex()
        # if len(self.plot.pre_intervals[idx]) == 0:
        #     QMessageBox.warning(self, "No selected data", "You have not selected any data")
            # return

        for rect in self.plot.pre_intervals[idx] + self.plot.line_region_item_list[idx]:
            rect.set_selected(False)
            self.plot.on_region_clicked(rect)
            # move the pre selected intervals are ready for further handlings
            # print(len(self.plot.pre_intervals[self.auto_pre_recid_combobox.currentIndex()]))

        # self.pre_selection_indicator_list[self.auto_pre_recid_combobox.currentIndex()] = not self.pre_selection_indicator_list[self.auto_pre_recid_combobox.currentIndex()]
        # print(f'self.pre_selection_indicator_list = {self.pre_selection_indicator_list}')

    def undo_pick_all(self):
        for rect in self.plot.picked_lr_items:
            rect.set_selected(False)
        self.plot.picked_lr_items.clear()
        print(len(self.plot.picked_lr_items))

    def on_line_return_pressed(self):
        self.pbar.reset()
        comment = self.line_edit.text()
        if self.plot.selected_interval is None or len(comment) == 0:
            QMessageBox.warning(self, "Alert!", "Null.")
            return

        recid_to_comment = self.combo.currentIndex()
        print(f"recid_to_comment = self.combo.currentIndex() {recid_to_comment}")

        checked_check_boxes = [idx for idx, box in enumerate(self.checkbox_list) if box.isChecked()]
        if recid_to_comment not in checked_check_boxes:
            QMessageBox.warning(self, "Alert Mistake!", "You are trying to add comment to the wrong time serie.")
            return

        self.pbar.setMinimum(0)
        self.pbar.setMaximum(2 * len(self.plot.picked_lr_items))
        self.store_selected_interval()  # update attributes of rects in self.plot.picked_lr_items
        for rect in self.plot.picked_lr_items:
            rect.set_comment(comment)

            # gui
            rect.set_exported()  # convert to exported rect -> grey
            rect.set_visible(self.plot.qt_checkboxes[rect.recid])
            if rect not in self.plot.to_export_rects[recid_to_comment]:
                self.plot.to_export_rects[recid_to_comment].append(rect)
                self.plot.to_export_rects_dict[recid_to_comment][self.spinbox.value()].append(rect)
                # print(f'self.plot.to_export_rects_dict[recid_to_comment][self.spinbox.value()]: {self.plot.to_export_rects_dict}')

            # data
            recid = rect.recid
            signal = rect.signal_array
            comment = rect.get_comment()
            start = rect.origin_start
            end = rect.origin_end
            self.dataset.output_df.add_row(recid=recid, signal=signal, comment=comment,
                                        file_path=Dataset.file_path[recid_to_comment], start=start, end=end)

            self.step += 1
            self.pbar.setValue(self.step)
            
        self.plot.picked_lr_items.clear()

        print(f'self.dataset.output_df.get_dataframe_list()')
        self.line_edit.clear()
        self.step = 0
        self.pbar.reset()

    def on_height_given(self):
        text = self.height_line_edit.text()
        if text is None:
            return

        try:
            text = text.replace(",", ".")
            cleaned_text = re.sub("[^0-9.]", "", text)
            if cleaned_text:
                self.height[self.auto_pre_recid_combobox.currentIndex()] = float(cleaned_text)
                self.sustained_thresholds_list[self.auto_pre_recid_combobox.currentIndex()] = float(cleaned_text)
            else:
                raise ValueError("Input is not a valid number!")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Could not convert '{text}' to float: {e}")

        # self.height = float(text)
        print(f"self.height = {self.height}")


    def on_distance_given(self):
        text = self.distance_line_edit.text()
        if text is None:
            return

        try:
            text = text.replace(",", ".")
            cleaned_text = re.sub("[^0-9.]", "", text)
            if cleaned_text:
                self.distance[self.auto_pre_recid_combobox.currentIndex()] = float(cleaned_text)
            else:
                raise ValueError("Input is not a valid number!")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Could not convert '{text}' to float: {e}")

        print(f"self.distance = {self.distance}")

    def on_window_size_given(self):
        text = self.window_size_line_edit.text()
        if text is None:
            return

        try:
            text = text.replace(",", ".")
            cleaned_text = re.sub("[^0-9.]", "", text)
            if cleaned_text:
                self.window_size_list[self.auto_pre_recid_combobox.currentIndex()] = int(cleaned_text)
            else:
                raise ValueError("Input is not a valid number!")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Could not convert '{text}' to float: {e}")

        print(f"self.sustained_thresholds_list = {self.window_size_list}")

    def value_changed(self, i):
        if i > self.dataset.chunk_cnt:
            raise ValueError(f"{i} is out of range! Values between 0 and {self.dataset.chunk_cnt} are alowed!")

    def go_to_chunk(self):
        # if self.chunk_idx_before_reload == 0:
        # self.chunk_idx = self.spinbox.value()
        Dataset.chunk_idx = self.spinbox.value()
        chunk_idx = self.spinbox.value()
        self.dataframe = self.dataset.create_sample_data(chunk_idx)

        self.plot.get_new_chunk(self.dataframe)
        # self.plot.plot_envelope(height=self.height, distance=self.distance, show_peaks=self.show_peaks_checkbox.isChecked())
        self.clear_plot()
        self.plot.remove_all_peaks_in_new_chunk()
        self.plot.remove_all_envelope_in_new_chunk()

        if self.plot.offset_indicator_item:
            self.plot.removeItem(self.plot.offset_indicator_item)
            self.plot.offset_indicator_item = None

        for idx, checkbox in enumerate(self.checkbox_list):
            for chunk, rect_list in self.plot.to_export_rects_dict[idx].items():
                for rect in rect_list:
                    rect.set_visible(checkbox.isChecked() and chunk == chunk_idx)

    def next_chunk(self):
        next_idx = self.spinbox.value() + 1
        if next_idx > self.dataset.chunk_cnt:
            return
        self.spinbox.setValue(next_idx)
        Dataset.chunk_idx += 1
        self.dataframe = self.dataset.create_sample_data(next_idx)
        self.plot.get_new_chunk(self.dataframe)
        self.clear_plot()
        self.plot.remove_all_peaks_in_new_chunk()
        self.plot.remove_all_envelope_in_new_chunk()

        if self.plot.offset_indicator_item:
            self.plot.removeItem(self.plot.offset_indicator_item)
            self.plot.removeItem(self.plot.offset_indicator_item_limit_line_0)
            self.plot.removeItem(self.plot.offset_indicator_item_limit_line_1)
            self.plot.offset_indicator_item = None
            self.plot.offset_indicator_item_limit_line_0 = None
            self.plot.offset_indicator_item_limit_line_1 = None

        for idx, checkbox in enumerate(self.checkbox_list):
            for chunk, rect_list in self.plot.to_export_rects_dict[idx].items():
                for rect in rect_list:
                    rect.set_visible(checkbox.isChecked() and chunk == next_idx)

    def last_chunk(self):
        last_idx = self.spinbox.value() - 1
        if last_idx < 0:
            return
        self.spinbox.setValue(last_idx)
        Dataset.chunk_idx -= 1
        self.dataframe = self.dataset.create_sample_data(last_idx)
        self.plot.get_new_chunk(self.dataframe)
        self.clear_plot()
        self.plot.remove_all_peaks_in_new_chunk()
        self.plot.remove_all_envelope_in_new_chunk()

        if self.plot.offset_indicator_item:
            self.plot.removeItem(self.plot.offset_indicator_item)
            self.plot.removeItem(self.plot.offset_indicator_item_limit_line_0)
            self.plot.removeItem(self.plot.offset_indicator_item_limit_line_1)
            self.plot.offset_indicator_item = None
            self.plot.offset_indicator_item_limit_line_0 = None
            self.plot.offset_indicator_item_limit_line_1 = None

        for idx, checkbox in enumerate(self.checkbox_list):
            for chunk, rect_list in self.plot.to_export_rects_dict[idx].items():
                for rect in rect_list:
                    rect.set_visible(checkbox.isChecked() and chunk == last_idx)

    def checkbox_state_changed(self, state):
        is_checked = state == 2
        for idx, checkbox in enumerate(self.checkbox_list):
            if checkbox == self.sender():
                is_checked = checkbox.isChecked()
                self.plot.qt_checkboxes[idx] = is_checked
                self.plot.toggle_line(self.plot.plot_data_item_list[idx], is_checked)
                self.plot.toggle_line(self.plot.line_envelope_list[idx], is_checked and self.show_peaks_checkbox.isChecked())
                self.plot.toggle_line(self.plot.line_ssm_envelope_list[idx], is_checked and self.show_peaks_checkbox.isChecked())
                self.plot.toggle_line(self.plot.line_rms_envelope_list[idx], is_checked and self.show_peaks_checkbox.isChecked())
                self.plot.toggle_line(self.plot.peaks_list[idx], is_checked and self.show_peaks_checkbox.isChecked())

            for rect in self.plot.pre_intervals[idx]:
                rect.setVisible(is_checked)

            for lr_item in self.plot.line_region_item_list[idx]:
                lr_item.setVisible(is_checked)

            # for rect_to_delete in self.plot.to_export_rects[idx]:
            #     rect_to_delete.setVisible(is_checked)

            # for rect_to_delete in self.plot.to_export_rects_dict[idx][self.spinbox.value()]:
            #     rect_to_delete.setVisible(is_checked)

            for chunk, rect_list in self.plot.to_export_rects_dict[idx].items():
                for rect in rect_list:
                    rect.setVisible(is_checked and chunk == self.spinbox.value())

    def on_offset_mode_change(self, state):
        is_checked = state == 2
        self.offset_mode_checkbox = is_checked
        for plot in self.plot.plot_data_item_list:
            plot.setCurveClickable(is_checked, width=5)

    def on_show_peaks(self, state=None):
        is_checked = state == 2
        for idx, bool_state in enumerate(self.plot.qt_checkboxes):
            self.plot.toggle_line(self.plot.line_envelope_list[idx], is_checked and bool_state)
            self.plot.toggle_line(self.plot.line_ssm_envelope_list[idx], is_checked and bool_state)
            self.plot.toggle_line(self.plot.line_rms_envelope_list[idx], is_checked and bool_state)
            
            bool_state = self.checkbox_list[idx].isChecked()
            self.plot.toggle_line(self.plot.peaks_list[idx], is_checked and bool_state)
            print(f'show peaks is checked: {is_checked}, line id {idx} is checked: {bool_state}')

def select_files(s=None):
    desktop_path = os.path.expanduser("~/Desktop")
    files, _ = QFileDialog.getOpenFileNames(None, 
                                            "Select initial files",
                                            desktop_path,
                                            # TODO set default path as desktop
                                            # "C:/Users/liwa/code/data",
                                            "Npy Files (*.npy);; Npz Files (*.npz);; All Files (*.*);;Text Files (*.txt)")
    if not files:
        QMessageBox.warning(None, "Warning!!", "You have to select files!")
        sys.exit()
    # files.sort()
    Dataset.file_path.extend(files)
    print(files)

class PlotThread(QThread):
    preselection_signal = pyqtSignal(object)
    
    def __init__(self):
        pass
    
    def run(self):
        pass

def main():
    # QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setStyle("macOS")
    app.setStyleSheet(StyleSheet)
    # font = QFont("Arial", 13)
    # app.setFont(font)
    # app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    select_files()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    # stats.dump_stats("profile_results.prof")

    # cProfile.run("main()", "restats")
