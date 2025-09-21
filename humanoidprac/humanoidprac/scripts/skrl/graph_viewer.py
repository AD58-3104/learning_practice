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

class GraphViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None
        self.data1 = None
        self.data2 = None
        self.current_file1 = None
        self.current_file2 = None
        self.prefix1 = ""
        self.prefix2 = "ref_"
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CSV Graph Viewer')
        self.setGeometry(100, 100, 1400, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Top controls
        controls_layout = QVBoxLayout()

        # File 1 selection
        file1_layout = QHBoxLayout()
        self.open_button1 = QPushButton('Open CSV File 1')
        self.open_button1.clicked.connect(lambda: self.open_file(1))
        file1_layout.addWidget(self.open_button1)

        file1_layout.addWidget(QLabel('Prefix:'))
        self.prefix1_edit = QLineEdit()
        self.prefix1_edit.setText(self.prefix1)
        self.prefix1_edit.setMaximumWidth(100)
        self.prefix1_edit.textChanged.connect(self.on_prefix_changed)
        file1_layout.addWidget(self.prefix1_edit)

        self.file_label1 = QLabel('No file loaded')
        file1_layout.addWidget(self.file_label1)
        file1_layout.addStretch()

        controls_layout.addLayout(file1_layout)

        # File 2 selection
        file2_layout = QHBoxLayout()
        self.open_button2 = QPushButton('Open CSV File 2')
        self.open_button2.clicked.connect(lambda: self.open_file(2))
        file2_layout.addWidget(self.open_button2)

        file2_layout.addWidget(QLabel('Prefix:'))
        self.prefix2_edit = QLineEdit()
        self.prefix2_edit.setText(self.prefix2)
        self.prefix2_edit.setMaximumWidth(100)
        self.prefix2_edit.textChanged.connect(self.on_prefix_changed)
        file2_layout.addWidget(self.prefix2_edit)

        self.file_label2 = QLabel('No file loaded')
        file2_layout.addWidget(self.file_label2)
        file2_layout.addStretch()

        controls_layout.addLayout(file2_layout)

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
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])

        main_layout.addWidget(splitter)

    def open_file(self, file_num):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Open CSV File {file_num}",
            "exp_logdata",
            "CSV Files (*.csv);;All Files (*.*)"
        )

        if file_path:
            try:
                data = pd.read_csv(file_path)
                file_name = Path(file_path).name

                if file_num == 1:
                    self.data1 = data
                    self.current_file1 = file_name
                    self.file_label1.setText(f"File 1: {file_name}")
                else:
                    self.data2 = data
                    self.current_file2 = file_name
                    self.file_label2.setText(f"File 2: {file_name}")

                self.update_combined_data()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def on_prefix_changed(self):
        self.prefix1 = self.prefix1_edit.text()
        self.prefix2 = self.prefix2_edit.text()
        self.update_combined_data()

    def update_combined_data(self):
        # Combine data from both files if they exist
        if self.data1 is not None or self.data2 is not None:
            combined_data = pd.DataFrame()

            # Add data from file 1 with prefix
            if self.data1 is not None:
                for col in self.data1.columns:
                    combined_data[self.prefix1 + col] = self.data1[col]

            # Add data from file 2 with prefix
            if self.data2 is not None:
                for col in self.data2.columns:
                    combined_data[self.prefix2 + col] = self.data2[col]

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
            x_label = "Index"
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
        if len(y_columns) == 1:
            ax.set_ylabel(y_columns[0])
        else:
            ax.set_ylabel("Value")

        # Create title showing loaded files
        title_parts = []
        if self.current_file1:
            title_parts.append(f"File1: {self.current_file1}")
        if self.current_file2:
            title_parts.append(f"File2: {self.current_file2}")

        if title_parts:
            title = " | ".join(title_parts)
        else:
            title = "Data"
        ax.set_title(f"{title} - {', '.join(y_columns)}")

        # Apply options
        if self.grid_checkbox.isChecked():
            ax.grid(True, alpha=0.3)

        if self.legend_checkbox.isChecked() and len(y_columns) > 1:
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