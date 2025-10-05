import os
from typing import Dict, Any


class CommandGenerator:
    """
    コマンド生成機能を提供するクラス。
    """

    def __init__(self):
        pass

    def build_command(self,
                     dir_path: str,
                     clicked_item_text: str,
                     command_template: str,
                     task_name: str,
                     json_params: Dict[str, Any],
                     record_video: bool = False,
                     headless: bool = False,
                     additional_args: str = "") -> str:
        """
        パラメータからコマンドを構築する。

        Args:
            dir_path: ディレクトリパス
            clicked_item_text: 選択されたアイテム名
            command_template: コマンドテンプレート
            task_name: タスク名
            json_params: jsonから読んだ追加パラメータの辞書
            record_video: ビデオ録画フラグ
            headless: ヘッドレスモードフラグ
            additional_args: 追加引数

        Returns:
            構築されたコマンド文字列
        """
        full_path = os.path.join(dir_path, clicked_item_text).replace('\\', '/')
        task_string = f"--task {task_name}"

        final_string = " ".join([command_template, full_path, task_string, self.parse_json_params(json_params)])

        # ビデオ撮影オプションを追加
        if record_video:
            final_string += " --video --video_length 1000"

        # ヘッドレスモードオプションを追加
        if headless:
            final_string += " --headless"

        # ユーザーが入力した追加引数を追加
        if additional_args.strip():
            final_string += " " + additional_args.strip()

        return final_string
    

    # jsonで指定されたパラメータをそれに応じたコマンドライン引数に変換する
    def parse_json_params(self, json_params : Dict[str, Any])-> str:
        if json_params.get('param_types', "") == "aaaa":
            return ""
        else:   # param_typesが無い場合はjoint_torque_limitとして扱う
            return self.joint_torque_limit_params(json_params)

    def joint_torque_limit_params(self, json_joint_params: Dict[str, Any]) -> str:
        """
        jsonから読んだジョイントパラメータから関節のトルク制限の文字列を生成する。
        change_joint_torqueイベントのパラメータと、ImplicitActuator側でのeffort_limitの設定を行う。

        Args:
            json_joint_params: ジョイントパラメータ辞書

        Returns:
            ジョイントパラメータ文字列
        """
        result_str = ""
        torques = json_joint_params.get('joint_torques', [])
        names = json_joint_params.get('joint_names', [])

        if not torques or not names:
            return result_str

        # トルク設定
        torque_str = str(torques).replace(' ', '')
        result_str += " " + f'env.events.change_joint_torque.params.joint_torque={torque_str}'

        # ジョイント名設定
        names_str = str(names).replace(' ', '').replace("'", '"')
        result_str += " " + f"'env.events.change_joint_torque.params.asset_cfg.joint_names={names_str}'"

        # ImplicitActuator側でのeffort_limitの設定
        effort_limit_str = "--joint_cfg '{"
        for joint_name, torque in zip(names, torques):
            effort_limit_str += f'"{joint_name}":{torque}, '
        effort_limit_str = effort_limit_str.rstrip(", ") + "}'"
        result_str += " " + effort_limit_str

        return result_str