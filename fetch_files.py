#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os
import paramiko
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QGroupBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont


class SSHWorker(QThread):
    output_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, host, port, username, password, command, private_key_path=None, passphrase=None):
        super().__init__()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.command = command
        self.private_key_path = private_key_path
        self.passphrase = passphrase

    def run(self):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if self.private_key_path and os.path.exists(self.private_key_path):
                # Use private key authentication
                try:
                    key = paramiko.RSAKey.from_private_key_file(self.private_key_path, password=self.passphrase)
                except:
                    # Try loading as DSA key if RSA fails
                    try:
                        key = paramiko.DSSKey.from_private_key_file(self.private_key_path, password=self.passphrase)
                    except:
                        # Try loading as ECDSA key
                        try:
                            key = paramiko.ECDSAKey.from_private_key_file(self.private_key_path, password=self.passphrase)
                        except:
                            # Try loading as Ed25519 key
                            key = paramiko.Ed25519Key.from_private_key_file(self.private_key_path, password=self.passphrase)
                ssh.connect(self.host, port=self.port, username=self.username, pkey=key)
            else:
                # Use password authentication
                ssh.connect(self.host, port=self.port, username=self.username, password=self.password)

            _, stdout, stderr = ssh.exec_command(self.command)
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')

            ssh.close()

            if error:
                self.error_occurred.emit(error)
            else:
                lines = output.strip().split('\n')
                self.output_ready.emit(lines)
        except Exception as e:
            self.error_occurred.emit(str(e))


class SCPWorker(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, host, port, username, password, remote_files, local_dir, private_key_path=None, passphrase=None):
        super().__init__()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.remote_files = remote_files
        self.local_dir = local_dir
        self.private_key_path = private_key_path
        self.passphrase = passphrase

    def run(self):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            if self.private_key_path and os.path.exists(self.private_key_path):
                # Use private key authentication
                try:
                    key = paramiko.RSAKey.from_private_key_file(self.private_key_path, password=self.passphrase)
                except:
                    # Try loading as DSA key if RSA fails
                    try:
                        key = paramiko.DSSKey.from_private_key_file(self.private_key_path, password=self.passphrase)
                    except:
                        # Try loading as ECDSA key
                        try:
                            key = paramiko.ECDSAKey.from_private_key_file(self.private_key_path, password=self.passphrase)
                        except:
                            # Try loading as Ed25519 key
                            key = paramiko.Ed25519Key.from_private_key_file(self.private_key_path, password=self.passphrase)
                ssh.connect(self.host, port=self.port, username=self.username, pkey=key)
            else:
                # Use password authentication
                ssh.connect(self.host, port=self.port, username=self.username, password=self.password)

            sftp = ssh.open_sftp()

            for remote_item in self.remote_files:
                if remote_item.endswith('/'):
                    # Directory copy
                    local_dir = os.path.join(self.local_dir, os.path.basename(remote_item.rstrip('/')))
                    self.progress.emit(f"Creating directory: {local_dir}")
                    self._copy_directory_recursive(sftp, remote_item.rstrip('/'), local_dir)
                else:
                    # File copy
                    local_file = os.path.join(self.local_dir, os.path.basename(remote_item))
                    self.progress.emit(f"Copying: {remote_item} -> {local_file}")
                    sftp.get(remote_item, local_file)
                    self.progress.emit(f"Completed: {os.path.basename(remote_item)}")

            sftp.close()
            ssh.close()
            self.completed.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _copy_directory_recursive(self, sftp, remote_dir, local_dir):
        """Recursively copy a directory from remote to local"""
        try:
            # Create local directory if it doesn't exist
            os.makedirs(local_dir, exist_ok=True)
            from pathlib import Path
            remote_dir = Path(remote_dir).as_posix()
            # List remote directory contents
            for item in sftp.listdir_attr(remote_dir):
                remote_path = remote_dir + '/' + item.filename
                local_path = os.path.join(local_dir, item.filename)
                remote_path = Path(remote_path).as_posix()

                if item.st_mode is not None and (item.st_mode & 0o40000):  # Directory
                    self.progress.emit(f"Creating directory: {local_path}")
                    self._copy_directory_recursive(sftp, remote_path, local_path)
                else:  # File
                    self.progress.emit(f"Copying file: {remote_path} -> {local_path}")
                    sftp.get(remote_path, local_path)
                    self.progress.emit(f"Completed: {item.filename}")
        except Exception as e:
            self.progress.emit(f"Error copying directory {remote_dir}: {str(e)}")


class SSHFileBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.current_path = "/home/satoshi/learning_practice/humanoidprac/humanoidprac/scripts/skrl"
        self.ssh_client = None
        self.private_key_path = "C:/Users/inoue/.ssh/localserver"
        self.initUI()

    def initUI(self):
        self.setWindowTitle("SSH File Browser")
        self.resize(1000, 700)

        main_layout = QVBoxLayout()

        # Connection group
        conn_group = QGroupBox("SSH Connection Settings")
        conn_layout = QVBoxLayout()

        # Host and Port
        host_layout = QHBoxLayout()
        host_layout.addWidget(QLabel("Host:"))
        self.host_input = QLineEdit()
        self.host_input.setText("192.168.1.54")
        host_layout.addWidget(self.host_input)

        host_layout.addWidget(QLabel("Port:"))
        self.port_input = QLineEdit()
        self.port_input.setText("52149")
        self.port_input.setMaximumWidth(80)
        host_layout.addWidget(self.port_input)
        conn_layout.addLayout(host_layout)

        # Username and Password
        cred_layout = QHBoxLayout()
        cred_layout.addWidget(QLabel("Username:"))
        self.username_input = QLineEdit()
        self.username_input.setText("satoshi")
        cred_layout.addWidget(self.username_input)

        cred_layout.addWidget(QLabel("Password:"))
        self.password_input = QLineEdit("3104")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("(Optional if using key)")
        cred_layout.addWidget(self.password_input)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_ssh)
        cred_layout.addWidget(self.connect_button)
        conn_layout.addLayout(cred_layout)

        # Private key selection
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Private Key:"))
        self.key_path_input = QLineEdit()
        self.key_path_input.setPlaceholderText("(Optional) Path to SSH private key")
        self.key_path_input.setText(self.private_key_path)
        default_key = os.path.expanduser("~/.ssh/local_server")
        if os.path.exists(default_key):
            self.key_path_input.setText(default_key)
        key_layout.addWidget(self.key_path_input)

        self.browse_key_button = QPushButton("Browse...")
        self.browse_key_button.clicked.connect(self.browse_private_key)
        key_layout.addWidget(self.browse_key_button)
        conn_layout.addLayout(key_layout)

        # Passphrase for encrypted keys
        passphrase_layout = QHBoxLayout()
        passphrase_layout.addWidget(QLabel("Key Passphrase:"))
        self.passphrase_input = QLineEdit("3104")
        self.passphrase_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.passphrase_input.setPlaceholderText("(Optional) Passphrase for encrypted key")
        passphrase_layout.addWidget(self.passphrase_input)
        conn_layout.addLayout(passphrase_layout)

        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)

        # File browser section
        browser_group = QGroupBox("File Browser")
        browser_layout = QVBoxLayout()

        # Current path display
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Current Path:"))
        self.path_label = QLabel("/")
        font = QFont()
        font.setBold(True)
        self.path_label.setFont(font)
        path_layout.addWidget(self.path_label)
        path_layout.addStretch()

        self.parent_button = QPushButton("Go to Parent Directory")
        self.parent_button.clicked.connect(self.go_to_parent)
        self.parent_button.setEnabled(False)
        path_layout.addWidget(self.parent_button)

        browser_layout.addLayout(path_layout)

        # Splitter for file list and info
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.file_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        splitter.addWidget(self.file_list)

        # Info panel
        info_widget = QWidget()
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("Selected Items:"))
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        info_widget.setLayout(info_layout)
        splitter.addWidget(info_widget)

        splitter.setSizes([700, 300])
        browser_layout.addWidget(splitter)

        browser_group.setLayout(browser_layout)
        main_layout.addWidget(browser_group)

        # Copy settings
        copy_group = QGroupBox("Copy Settings")
        copy_layout = QVBoxLayout()

        dest_layout = QHBoxLayout()
        dest_layout.addWidget(QLabel("Destination Directory:"))
        self.dest_input = QLineEdit()
        self.dest_input.setText(os.path.expanduser("~/"))
        dest_layout.addWidget(self.dest_input)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_destination)
        dest_layout.addWidget(self.browse_button)
        copy_layout.addLayout(dest_layout)

        # Copy button
        self.copy_button = QPushButton("Copy Selected Items")
        self.copy_button.clicked.connect(self.copy_files)
        self.copy_button.setEnabled(False)
        copy_layout.addWidget(self.copy_button)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        copy_layout.addWidget(self.status_text)

        copy_group.setLayout(copy_layout)
        main_layout.addWidget(copy_group)

        self.setLayout(main_layout)

    def browse_private_key(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SSH Private Key",
            os.path.expanduser("~/.ssh"),
            "All Files (*)"
        )
        if file_path:
            self.key_path_input.setText(file_path)

    def connect_ssh(self):
        host = self.host_input.text()
        port = int(self.port_input.text())
        username = self.username_input.text()
        password = self.password_input.text()
        self.private_key_path = self.key_path_input.text()
        self.passphrase = self.passphrase_input.text()

        if not host or not username:
            QMessageBox.warning(self, "Warning", "Please fill in host and username")
            return

        if not self.private_key_path and not password:
            QMessageBox.warning(self, "Warning", "Please provide either a password or a private key")
            return

        self.status_text.append(f"Connecting to {username}@{host}:{port}...")

        # Test connection and get initial file list
        self.list_files()

    def list_files(self):
        host = self.host_input.text()
        port = int(self.port_input.text())
        username = self.username_input.text()
        password = self.password_input.text()

        command = f"ls -la {self.current_path}"

        self.worker = SSHWorker(host, port, username, password, command, self.private_key_path, self.passphrase)
        self.worker.output_ready.connect(self.update_file_list)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def update_file_list(self, lines):
        self.file_list.clear()
        self.status_text.append(f"Connected! Listing files in: {self.current_path}")
        self.path_label.setText(self.current_path)
        self.connect_button.setText("Refresh")
        self.parent_button.setEnabled(self.current_path != "/")
        self.copy_button.setEnabled(True)

        for line in lines:
            if line and not line.startswith("total"):
                parts = line.split(None, 8)
                if len(parts) >= 9:
                    permissions = parts[0]
                    filename = parts[8]

                    # Skip . and .. entries
                    if filename in ['.', '..']:
                        continue

                    item = QListWidgetItem()

                    if permissions.startswith('d'):
                        item.setText(f"📁 {filename}")
                        item.setData(Qt.ItemDataRole.UserRole, ('dir', filename))
                    elif permissions.startswith('l'):
                        item.setText(f"🔗 {filename}")
                        item.setData(Qt.ItemDataRole.UserRole, ('link', filename))
                    else:
                        size = parts[4]
                        item.setText(f"📄 {filename} ({size})")
                        item.setData(Qt.ItemDataRole.UserRole, ('file', filename))

                    self.file_list.addItem(item)

        self.file_list.itemSelectionChanged.connect(self.update_selection_info)

    def update_selection_info(self):
        selected_items = self.file_list.selectedItems()
        info_lines = []
        for item in selected_items:
            file_type, filename = item.data(Qt.ItemDataRole.UserRole)
            full_path = os.path.join(self.current_path, filename)
            info_lines.append(f"{file_type.upper()}: {full_path}")
        self.info_text.setText("\n".join(info_lines))

    def on_item_double_clicked(self, item):
        file_type, filename = item.data(Qt.ItemDataRole.UserRole)

        if file_type == 'dir':
            # Navigate to the directory
            if self.current_path.endswith('/'):
                self.current_path = self.current_path + filename
            else:
                self.current_path = self.current_path + '/' + filename

            self.list_files()

    def go_to_parent(self):
        if self.current_path != '/':
            self.current_path = os.path.dirname(self.current_path)
            if not self.current_path:
                self.current_path = '/'
            self.list_files()

    def browse_destination(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Directory",
            self.dest_input.text()
        )
        if directory:
            self.dest_input.setText(directory)

    def copy_files(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No items selected")
            return

        remote_items = []
        for item in selected_items:
            file_type, filename = item.data(Qt.ItemDataRole.UserRole)
            full_path = os.path.join(self.current_path, filename)

            if file_type == 'dir':
                # Mark directories with trailing slash for recursive copy
                remote_items.append(full_path + '/')
            elif file_type == 'file':
                remote_items.append(full_path)

        if not remote_items:
            QMessageBox.warning(self, "Warning", "No valid items selected")
            return

        local_dir = self.dest_input.text()
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        host = self.host_input.text()
        port = int(self.port_input.text())
        username = self.username_input.text()
        password = self.password_input.text()

        self.copy_worker = SCPWorker(host, port, username, password, remote_items, local_dir, self.private_key_path, self.passphrase)
        self.copy_worker.progress.connect(self.update_copy_progress)
        self.copy_worker.completed.connect(self.copy_completed)
        self.copy_worker.error_occurred.connect(self.handle_error)
        self.copy_button.setEnabled(False)
        self.copy_worker.start()

    def update_copy_progress(self, message):
        self.status_text.append(message)
        # Auto scroll to bottom
        cursor = self.status_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.status_text.setTextCursor(cursor)

    def copy_completed(self):
        self.status_text.append("All items copied successfully!")
        self.copy_button.setEnabled(True)
        QMessageBox.information(self, "Success", "Items copied successfully!")

    def handle_error(self, error_msg):
        self.status_text.append(f"Error: {error_msg}")
        self.copy_button.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SSHFileBrowser()
    window.show()
    sys.exit(app.exec())