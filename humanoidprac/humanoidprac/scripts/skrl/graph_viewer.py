#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QComboBox,
                             QLabel, QSplitter, QListWidget, QCheckBox,
                             QSpinBox, QGroupBox, QMessageBox, QLineEdit,
                             QTabWidget, QTextEdit)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent, default_filename="graph.png"):
        super().__init__(canvas, parent)
        self.default_filename = default_filename

    def set_default_filename(self, filename):
        self.default_filename = filename

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = self.default_filename

        filters = []
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filters.append(f'{name} ({exts_list})')
        filters = ';;'.join(filters)

        fname, _ = QFileDialog.getSaveFileName(
            self.canvas.parent(), "Choose a filename to save to",
            startpath, filters)
        if fname:
            try:
                self.canvas.figure.savefig(fname)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QMessageBox.StandardButton.Ok)

class GraphViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.data_files = [None] * 6  # 最大6ファイル
        self.current_files = [None] * 6
        self.prefixes = ["", "ref_", "f3_", "f4_", "f5_", "f6_"]
        self.last_directory = "exp_logdata"
        self.default_save_filename = "graph.png"  # デフォルトの保存ファイル名

        # 予め指定しておく選択グループ（グループ名: {"columns": [カラム名のリスト], "ylabel": Y軸ラベル}）
        self.preset_groups = {
            "Group 1": {
                "columns": ["left_hip_yaw", "right_hip_yaw",
                            "right_hip_roll","left_hip_roll",
                            "left_hip_pitch","right_hip_pitch",
                            "left_knee","right_knee",
                            "left_ankle","right_ankle"],
                "ylabel": "Torque [Nm]"
            },
            "Group 2": {
                "columns": ["lin_vel_xy_yaw", "ang_vel_z_world"],
                "ylabel": "Value"
            },
            "Group 3": {
                "columns": ["column6", "column7", "column8", "column9"],
                "ylabel": "Group 3 Values"
            },
        }

        self.initUI()

    def initUI(self):
        self.setWindowTitle('CSV Graph Viewer')
        self.setGeometry(100, 100, 1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top controls
        controls_layout = QVBoxLayout()

        # File selection (最大6ファイル)
        self.file_buttons = []
        self.prefix_edits = []
        self.file_labels = []

        for i in range(6):
            file_layout = QHBoxLayout()

            # Open button
            open_button = QPushButton(f'Open CSV File {i+1}')
            open_button.clicked.connect(lambda checked, idx=i: self.open_file(idx))
            file_layout.addWidget(open_button)
            self.file_buttons.append(open_button)

            # Prefix
            file_layout.addWidget(QLabel('Prefix:'))
            prefix_edit = QLineEdit()
            prefix_edit.setText(self.prefixes[i])
            prefix_edit.setMaximumWidth(100)
            prefix_edit.textChanged.connect(lambda text, idx=i: self.on_prefix_changed(idx, text))
            file_layout.addWidget(prefix_edit)
            self.prefix_edits.append(prefix_edit)

            # File label
            file_label = QLabel('No file loaded')
            file_layout.addWidget(file_label)
            file_layout.addStretch()
            self.file_labels.append(file_label)

            controls_layout.addLayout(file_layout)

        # Plot type selection
        plot_type_layout = QHBoxLayout()
        plot_type_layout.addWidget(QLabel('Plot Type:'))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['Line', 'Scatter', 'Bar', 'Histogram'])
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)
        plot_type_layout.addWidget(self.plot_type_combo)
        plot_type_layout.addStretch()

        controls_layout.addLayout(plot_type_layout)

        main_layout.addLayout(controls_layout)

        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel with tabs
        left_panel = QTabWidget()

        # Tab 1: Column Selection and Options
        selection_tab = QWidget()
        selection_layout = QVBoxLayout(selection_tab)

        # Column selection group
        column_group = QGroupBox("Select Columns")
        column_layout = QVBoxLayout()

        column_layout.addWidget(QLabel("X-Axis:"))
        self.x_column_list = QListWidget()
        self.x_column_list.itemClicked.connect(self.update_plot)
        column_layout.addWidget(self.x_column_list)

        column_layout.addWidget(QLabel("Y-Axis (multi-select):"))
        self.y_column_list = QListWidget()
        self.y_column_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.y_column_list.itemSelectionChanged.connect(self.update_plot)
        column_layout.addWidget(self.y_column_list)

        # Preset selection buttons
        preset_label = QLabel("Quick Select:")
        column_layout.addWidget(preset_label)

        preset_buttons_layout = QVBoxLayout()
        for group_name in self.preset_groups.keys():
            btn = QPushButton(group_name)
            btn.clicked.connect(lambda _, name=group_name: self.select_preset_group(name))
            preset_buttons_layout.addWidget(btn)
        column_layout.addLayout(preset_buttons_layout)

        column_group.setLayout(column_layout)
        selection_layout.addWidget(column_group)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.grid_checkbox)

        self.legend_checkbox = QCheckBox("Show Legend")
        self.legend_checkbox.setChecked(True)
        self.legend_checkbox.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.legend_checkbox)

        # Sampling controls
        sampling_layout = QHBoxLayout()
        sampling_layout.addWidget(QLabel("Sample every:"))
        self.sample_spinbox = QSpinBox()
        self.sample_spinbox.setMinimum(1)
        self.sample_spinbox.setMaximum(1000)
        self.sample_spinbox.setValue(1)
        self.sample_spinbox.valueChanged.connect(self.update_plot)
        sampling_layout.addWidget(self.sample_spinbox)
        sampling_layout.addWidget(QLabel("points"))
        options_layout.addLayout(sampling_layout)

        # Y-axis label controls
        ylabel_layout = QHBoxLayout()
        ylabel_layout.addWidget(QLabel("Y-axis label:"))
        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.setPlaceholderText("Auto")
        self.ylabel_edit.textChanged.connect(self.update_plot)
        ylabel_layout.addWidget(self.ylabel_edit)
        options_layout.addLayout(ylabel_layout)

        # Save filename controls
        save_filename_layout = QHBoxLayout()
        save_filename_layout.addWidget(QLabel("Save filename:"))
        self.save_filename_edit = QLineEdit()
        self.save_filename_edit.setText(self.default_save_filename)
        self.save_filename_edit.textChanged.connect(self.on_save_filename_changed)
        save_filename_layout.addWidget(self.save_filename_edit)
        options_layout.addLayout(save_filename_layout)

        options_group.setLayout(options_layout)
        selection_layout.addWidget(options_group)

        selection_layout.addStretch()

        # Tab 2: Statistics
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)

        # Add tabs to the tab widget
        left_panel.addTab(selection_tab, "Selection")
        left_panel.addTab(stats_tab, "Statistics")

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Matplotlib Figure and Canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Navigation toolbar
        self.toolbar = CustomNavigationToolbar(self.canvas, self, self.default_save_filename)
        right_layout.addWidget(self.toolbar)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])

        main_layout.addWidget(splitter)

    def open_file(self, file_idx):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Open CSV File {file_idx+1}",
            self.last_directory,
            "CSV Files (*.csv);;All Files (*.*)"
        )

        if file_path:
            try:
                data = pd.read_csv(file_path)
                file_name = Path(file_path).name
                self.last_directory = str(Path(file_path).parent)

                self.data_files[file_idx] = data
                self.current_files[file_idx] = file_name
                self.file_labels[file_idx].setText(f"File {file_idx+1}: {file_name}")

                self.update_combined_data()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def on_prefix_changed(self, idx, text):
        self.prefixes[idx] = text
        self.update_combined_data()

    def on_save_filename_changed(self):
        self.default_save_filename = self.save_filename_edit.text()
        self.toolbar.set_default_filename(self.default_save_filename)

    def select_preset_group(self, group_name):
        """指定されたプリセットグループの項目を選択"""
        if group_name not in self.preset_groups:
            return

        preset = self.preset_groups[group_name]
        columns_to_select = preset["columns"]
        ylabel = preset.get("ylabel", "")

        # 全ての選択を解除
        self.y_column_list.clearSelection()

        # プリセットグループの項目を選択
        for i in range(self.y_column_list.count()):
            item = self.y_column_list.item(i)
            if item.text() in columns_to_select:
                item.setSelected(True)

        # Y軸ラベルを設定
        self.ylabel_edit.setText(ylabel)

    def update_combined_data(self):
        # Combine data from all loaded files
        has_data = any(data is not None for data in self.data_files)

        if has_data:
            combined_data = pd.DataFrame()

            # Add data from each file with its prefix
            for i, data in enumerate(self.data_files):
                if data is not None:
                    for col in data.columns:
                        combined_data[self.prefixes[i] + col] = data[col]

            self.data = combined_data

            # Update column lists
            columns = list(self.data.columns)

            self.x_column_list.clear()
            self.y_column_list.clear()

            # Add index as an option for x-axis
            self.x_column_list.addItem("Index")
            self.x_column_list.addItems(columns)
            self.y_column_list.addItems(columns)

            # Auto-select first items
            if self.x_column_list.count() > 0:
                self.x_column_list.setCurrentRow(0)
            if self.y_column_list.count() > 0:
                item = self.y_column_list.item(0)
                if item:
                    item.setSelected(True)

            self.update_plot()

    def update_plot(self):
        if self.data is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Get selected columns
        x_item = self.x_column_list.currentItem()
        if not x_item:
            return

        x_column = x_item.text()

        y_items = self.y_column_list.selectedItems()
        if not y_items:
            return

        y_columns = [item.text() for item in y_items]

        # Prepare x data
        if x_column == "Index":
            x_data = np.arange(len(self.data))
            x_label = "Step"
        else:
            x_data = np.asarray(self.data[x_column].values)
            x_label = x_column

        # Apply sampling
        sample_rate = self.sample_spinbox.value()
        x_data = x_data[::sample_rate]

        # Plot based on type
        plot_type = self.plot_type_combo.currentText()

        stats_text = "Statistics:\n"

        for y_column in y_columns:
            y_data = np.asarray(self.data[y_column].values)[::sample_rate]

            if plot_type == 'Line':
                ax.plot(x_data, y_data, label=y_column, alpha=0.8)
            elif plot_type == 'Scatter':
                ax.scatter(x_data, y_data, label=y_column, alpha=0.6, s=10)
            elif plot_type == 'Bar':
                if len(y_columns) == 1:
                    ax.bar(x_data[:50], y_data[:50], label=y_column, alpha=0.8)
                else:
                    width = 0.8 / len(y_columns)
                    offset = np.arange(len(y_columns)) * width - 0.4 + width/2
                    ax.bar(x_data[:50] + offset[y_columns.index(y_column)],
                          y_data[:50], width, label=y_column, alpha=0.8)
            elif plot_type == 'Histogram':
                ax.hist(y_data, bins=50, label=y_column, alpha=0.6)

            # Calculate statistics
            stats_text += f"\n{y_column}:\n"
            stats_text += f"  Mean: {np.mean(y_data):.4f}\n"
            stats_text += f"  Std: {np.std(y_data):.4f}\n"
            stats_text += f"  Min: {np.min(y_data):.4f}\n"
            stats_text += f"  Max: {np.max(y_data):.4f}\n"

        self.stats_text.setText(stats_text)

        # Set labels and title
        ax.set_xlabel(x_label)

        # Set Y-axis label (custom or auto)
        custom_ylabel = self.ylabel_edit.text().strip()
        if custom_ylabel:
            ax.set_ylabel(custom_ylabel)
        elif len(y_columns) == 1:
            ax.set_ylabel(y_columns[0])
        else:
            ax.set_ylabel("Value")


        # Apply options
        if self.grid_checkbox.isChecked():
            ax.grid(True, alpha=0.3)

        if self.legend_checkbox.isChecked():
            ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    viewer = GraphViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()