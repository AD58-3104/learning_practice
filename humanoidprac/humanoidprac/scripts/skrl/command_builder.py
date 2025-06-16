# 次の要件を満たすPyQtアプリを書いて下さい。ユーザが入力可能なLineEditは2つ。1つはディレクトリへの相対パスを指定する、もう1つは実行するコマンドのひな形を入力する。ディレクトリのLineEditにユーザが入力すると、そのディレクトリ内の要素が一覧表示される。その一覧表示のうち1つの要素をクリックすると、相対パス + 要素 + コマンドのひな形が結合された文字列をクリップボードにコピーできる。

import sys
import os
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QLabel, QTextEdit, QPushButton
)
from PyQt6.QtCore import Qt

class CommandBuilderApp(QWidget):
    """
    ディレクトリ内容の表示、コマンド作成、ファイルプレビュー機能を持つ
    PyQtアプリケーション。
    """
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # --- ウィンドウの基本設定 ---
        self.setWindowTitle('コマンドビルダー & ファイルプレビュー')
        # ウィンドウサイズを広げる
        self.setGeometry(200, 200, 1000, 500)

        # --- メインレイアウト（左右分割） ---
        main_layout = QHBoxLayout()

        # --- 左側パネルのレイアウト ---
        left_panel_layout = QVBoxLayout()

        # 1. ディレクトリパス入力
        left_panel_layout.addWidget(QLabel('📂 相対パス (Relative Path):'))
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setText('./logs/skrl/h1_flat/joint_experiment')
        self.dir_path_edit.textChanged.connect(self.update_directory_list)
        left_panel_layout.addWidget(self.dir_path_edit)

        # 2. コマンドひな形入力
        left_panel_layout.addWidget(QLabel('⚙️ コマンドひな形 (Command Template):'))
        self.command_template_edit = QLineEdit()
        self.command_template_edit.setText('./run_play.sh ')
        left_panel_layout.addWidget(self.command_template_edit)

        # 3. ★追加: プレビュー用ファイル名入力
        left_panel_layout.addWidget(QLabel('📄 プレビューファイル名 (Preview File in Dir):'))
        self.preview_file_edit = QLineEdit()
        self.preview_file_edit.setText('joint_cfg.json など (フォルダ選択時に使用)')
        left_panel_layout.addWidget(self.preview_file_edit)

        # 4. ファイル/ディレクトリ一覧
        left_panel_layout.addWidget(QLabel('📋 ファイル/ディレクトリ一覧 (クリックしてコピー/プレビュー):'))
        self.file_list_widget = QListWidget()
        # self.file_list_widget.itemClicked.connect(self.copy_to_clipboard)
        self.file_list_widget.currentItemChanged.connect(self.copy_to_clipboard)
        self.file_list_widget.currentItemChanged.connect(self.display_file_content)
        left_panel_layout.addWidget(self.file_list_widget)

        self.exec_button = QPushButton("Execute")
        self.exec_button.clicked.connect(self.execute_current_command)
        left_panel_layout.addWidget(self.exec_button)

        # 左側パネルをウィジェットとしてまとめる
        left_widget = QWidget()
        left_widget.setLayout(left_panel_layout)

        # 右側パネルのレイアウト
        right_panel_layout = QVBoxLayout()

        # --- ★追加: 右側パネル（テキストプレビュー） ---
        self.preview_text_edit = QTextEdit()
        self.preview_text_edit.setReadOnly(True) # 読み取り専用にする
        self.preview_text_edit.setPlaceholderText("左のリストからファイルをクリックすると、ここに内容が表示されます。")
        right_panel_layout.addWidget(self.preview_text_edit)

        self.command_preview_line_edit = QLineEdit()
        self.command_preview_line_edit.setReadOnly(True)
        right_panel_layout.addWidget(self.command_preview_line_edit)

        # 右側ウィジェット
        right_widget = QWidget()
        right_widget.setLayout(right_panel_layout)


        # --- メインレイアウトに左右のパネルを追加 ---
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        # 左右のパネルの幅の比率を設定 (1:1)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)

        self.setLayout(main_layout)

        # --- 初期表示 ---
        self.update_directory_list(self.dir_path_edit.text())

    def update_directory_list(self, path: str):
        self.file_list_widget.clear()
        if os.path.isdir(path):
            try:
                entries = sorted(os.listdir(path))
                for index, entry in enumerate(entries):
                    display_text = f"{index}: {entry}"
                    self.file_list_widget.addItem(display_text)
            except OSError as e:
                self.file_list_widget.addItem(f"エラー: {e}")
        else:
            self.file_list_widget.addItem("有効なディレクトリではありません。")

    def copy_to_clipboard(self, item: QListWidgetItem):
        if not item:
            return
        display_text = item.text()
        parts = display_text.split(': ', 1)
        if len(parts) == 2 and parts[0].isdigit():
            clicked_item_text = parts[1]
        else:
            return

        dir_path = self.dir_path_edit.text()
        command_template = self.command_template_edit.text()
        full_path = os.path.join(dir_path, clicked_item_text)
        final_string = command_template + full_path.replace('\\', '/')
        clipboard = QApplication.clipboard()
        clipboard.setText(final_string)
        self.command_preview_line_edit.setText(final_string)
        print(f"クリップボードにコピーしました: {final_string}")

    def display_file_content(self, current_item: QListWidgetItem, previous_item: QListWidgetItem):
        """
        選択されたアイテム(current_item)に応じてファイル内容をプレビュー表示する。
        """
        # 選択が外れた場合など、current_itemがNoneになることがあるためチェック
        if not current_item:
            self.preview_text_edit.setPlaceholderText("左のリストからファイルを選択してください。")
            self.preview_text_edit.clear()
            return

        display_text = current_item.text()

        parts = display_text.split(': ', 1)

        if not (len(parts) == 2 and parts[0].isdigit()):
            self.preview_text_edit.setText("プレビューできません。（エラー項目）")
            return

        item_name = parts[1]
        base_path = self.dir_path_edit.text()
        path_to_check = os.path.join(base_path, item_name)
        preview_target_path = ""

        # クリックされたのがディレクトリかファイルか判定
        if os.path.isdir(path_to_check):
            # ディレクトリの場合、プレビュー用LineEditからファイル名を取得
            preview_filename = self.preview_file_edit.text()
            if not preview_filename:
                self.preview_text_edit.setText(f"ディレクトリ '{item_name}' が選択されました。\nプレビューしたいファイル名を上の欄に入力してください。")
                return
            preview_target_path = os.path.join(path_to_check, preview_filename)
        elif os.path.isfile(path_to_check):
            # ファイルの場合、それがプレビュー対象
            preview_target_path = path_to_check
        else:
            self.preview_text_edit.setText(f"'{item_name}' はプレビュー対象外です（シンボリックリンクなど）。")
            return

        # ファイル内容を読み込んで表示
        if os.path.isfile(preview_target_path):
            try:
                with open(preview_target_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.preview_text_edit.setText(content)
                print(f"Debug content {preview_target_path}")
            except UnicodeDecodeError:
                self.preview_text_edit.setText(f"エラー:\nファイル '{os.path.basename(preview_target_path)}' はUTF-8でデコードできませんでした。")
            except Exception as e:
                self.preview_text_edit.setText(f"ファイル読み込みエラー:\n{e}")
        else:
            self.preview_text_edit.setText(f"ファイルが見つかりません:\n{preview_target_path}")

    def execute_current_command(self):
        command = "gnome-terminal -- " + self.command_preview_line_edit.text()
        print(f"Execute command:: {command}")
        subprocess.run(command,shell=True)

def main():
    app = QApplication(sys.argv)
    ex = CommandBuilderApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()