# 次の要件を満たすPyQtアプリを書いて下さい。ユーザが入力可能なLineEditは2つ。1つはディレクトリへの相対パスを指定する、もう1つは実行するコマンドのひな形を入力する。ディレクトリのLineEditにユーザが入力すると、そのディレクトリ内の要素が一覧表示される。その一覧表示のうち1つの要素をクリックすると、相対パス + 要素 + コマンドのひな形が結合された文字列をクリップボードにコピーできる。

import sys
import os
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QLabel, QTextEdit, QPushButton, QCheckBox
)
from PyQt6.QtCore import Qt
import json

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
        self.dir_path_edit.setText('./logs/skrl/h1_flat/joint_experiment_ver3')
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
        self.preview_file_edit.setText('joint_cfg.json')
        left_panel_layout.addWidget(self.preview_file_edit)

        # 5. ファイル再読み込みボタン
        self.reload_button = QPushButton("🔄 ファイルを再読み込み")
        self.reload_button.clicked.connect(self.reload_current_file)
        left_panel_layout.addWidget(self.reload_button)

        # 4. ファイル/ディレクトリ一覧
        left_panel_layout.addWidget(QLabel('📋 ファイル/ディレクトリ一覧 (クリックしてコピー/プレビュー):'))
        self.file_list_widget = QListWidget()
        # self.file_list_widget.itemClicked.connect(self.copy_to_clipboard)
        self.file_list_widget.currentItemChanged.connect(self.display_file_content)
        # self.file_list_widget.currentItemChanged.connect(self.copy_to_clipboard)
        self.current_joint_params = {}
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
        self.preview_text_edit.setReadOnly(False)
        self.preview_text_edit.setPlaceholderText("左のリストからファイルをクリックすると、ここに内容が表示されます。")
        self.preview_text_edit.textChanged.connect(self.on_preview_text_changed)
        right_panel_layout.addWidget(self.preview_text_edit)

        # ビデオ撮影とヘッドレスモードのチェックボックスを横並びに配置
        checkbox_layout = QHBoxLayout()
        
        self.record_video_checkbox = QCheckBox("🎥 ビデオを撮影する (Record Video)")
        self.record_video_checkbox.stateChanged.connect(self.on_checkbox_changed)
        checkbox_layout.addWidget(self.record_video_checkbox)
        
        self.headless_checkbox = QCheckBox("🖥️ ヘッドレスモード (Headless)")
        self.headless_checkbox.stateChanged.connect(self.on_checkbox_changed)
        checkbox_layout.addWidget(self.headless_checkbox)
        
        checkbox_widget = QWidget()
        checkbox_widget.setLayout(checkbox_layout)
        right_panel_layout.addWidget(checkbox_widget)

        # タスク指定用のラベルとLineEdit
        right_panel_layout.addWidget(QLabel('🔧 実行タスク (Execution Task):'))
        self.execute_task_edit = QLineEdit()
        self.execute_task_edit.setText("Humanoidprac-v0-play")
        self.execute_task_edit.textChanged.connect(self.on_execute_task_changed)
        right_panel_layout.addWidget(self.execute_task_edit)

        # 追加引数入力用のラベルとLineEdit
        right_panel_layout.addWidget(QLabel('🔧 追加引数 (Additional Arguments):'))
        self.additional_args_edit = QLineEdit()
        self.additional_args_edit.setPlaceholderText("例: --param1 value1 --param2 value2")
        self.additional_args_edit.textChanged.connect(self.on_additional_args_changed)
        right_panel_layout.addWidget(self.additional_args_edit)

        # コマンドプレビュー表示用
        right_panel_layout.addWidget(QLabel('📋 コマンドプレビュー:'))
        self.command_preview_line_edit = QLineEdit()
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

    def on_checkbox_changed(self, state):
        """チェックボックスの状態変更時の処理"""
        # 現在選択されているアイテムで再度コマンドを更新
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.copy_to_clipboard(current_item, None)

    def on_additional_args_changed(self, text):
        """追加引数が変更されたときの処理"""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.copy_to_clipboard(current_item, None)

    def on_execute_task_changed(self, text):
        """実行タスクが変更されたときの処理"""
        current_item = self.file_list_widget.currentItem()
        if current_item:
            self.copy_to_clipboard(current_item, None)

    def copy_to_clipboard(self, item: QListWidgetItem, previous_item: QListWidgetItem):
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
        task_string = f"--task {self.execute_task_edit.text()}"
        final_string = command_template + full_path.replace('\\', '/') + " " + task_string + " " + self.get_joint_parameter_strings()
        
        # ビデオ撮影オプションを追加
        if self.record_video_checkbox.isChecked():
            final_string += " --video --video_length 1000"
        
        # ヘッドレスモードオプションを追加
        if self.headless_checkbox.isChecked():
            final_string += " --headless"

        # ユーザーが入力した追加引数を追加
        additional_args = self.additional_args_edit.text().strip()
        if additional_args:
            final_string += " " + additional_args

        clipboard = QApplication.clipboard()
        clipboard.setText(final_string)
        self.command_preview_line_edit.setText(final_string)
        print(f"クリップボードにコピーしました: {final_string}")


    def get_joint_parameter_strings(self) -> str:
        result_str = ""
        torques = self.current_joint_params.get('joint_torques',[])
        torque_str = str(torques).replace(' ', '')  # スペースを除去
        result_str =  result_str + " " + (f'env.events.change_joint_torque.params.joint_torque={torque_str}')
        print(torque_str)

        # ジョイント名設定
        names = self.current_joint_params.get('joint_names',[])
        names_str = str(names).replace(' ', '')  # スペースを除去
        names_str = names_str.replace("'", '"')
        result_str = result_str + " " + (f"'env.events.change_joint_torque.params.asset_cfg.joint_names={names_str}'")
        print(names_str)
        # ↑なんかこれシングルクオートで囲って、]はエスケープしない。文字列は""で囲むでいけた。よくわからん。
        
        # ImplicitActuator側でのeffort_limitの設定
        effort_limit_str = "--joint_cfg '{"
        for joint_name, torque in zip(names, torques):
            effort_limit_str += f'"{joint_name}":{torque}, '
        effort_limit_str = effort_limit_str.rstrip(", ") + "}'"
        result_str = result_str + " " + effort_limit_str
        print(effort_limit_str)
        return result_str

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
            ext = os.path.splitext(preview_target_path)[1].lower()
            try:
                with open(preview_target_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # JSONファイルならパースしてcurrent_joint_paramsに保存
                    if ext in [".json"]:
                        self.current_joint_params = json.loads(content)
                self.preview_text_edit.setText(content)
                print(f"Debug content {preview_target_path}")
            except UnicodeDecodeError:
                self.preview_text_edit.setText(f"エラー:\nファイル '{os.path.basename(preview_target_path)}' はUTF-8でデコードできませんでした。")
            except Exception as e:
                self.preview_text_edit.setText(f"ファイル読み込みエラー:\n{e}")
        else:
            self.preview_text_edit.setText(f"ファイルが見つかりません:\n{preview_target_path}")
        
        self.copy_to_clipboard(current_item, previous_item)

    def on_preview_text_changed(self):
        """preview_text_editの内容が変更されたときの処理"""
        try:
            # JSONパースを試行
            content = self.preview_text_edit.toPlainText()
            if content.strip():
                self.current_joint_params = json.loads(content)
                # 現在選択されているアイテムでコマンドを更新
                current_item = self.file_list_widget.currentItem()
                if current_item:
                    self.copy_to_clipboard(current_item, None)
        except json.JSONDecodeError:
            # JSONでない場合は何もしない
            pass

    def execute_current_command(self):
        command = "gnome-terminal -- " + self.command_preview_line_edit.text()
        import datetime
        currnet_time = datetime.datetime.now()
        print(f"Execute command [{currnet_time}]: ")
        print(f"{command}")
        subprocess.run(command,shell=True)

    def reload_current_file(self):
        """ディレクトリ一覧を再読み込み"""
        self.update_directory_list(self.dir_path_edit.text())

def main():
    app = QApplication(sys.argv)
    ex = CommandBuilderApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()