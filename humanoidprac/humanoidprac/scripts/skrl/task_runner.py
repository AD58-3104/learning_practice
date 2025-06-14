#!/usr/bin/env python3

# コマンドメモ
# labpython train.py --task Humanoidprac-v0 --headless 
# agent.agent.experiment.directory="h1_flat/ex 
# 'env.events.change_joint_torque.params.joint_torque=[400.0]'
# 'env.events.change_joint_torque.params.asset_cfg.joint_names=["right_hip_roll"]'

"""
タスクランナー: YAML設定ファイルから複数の実験設定を読み込んで順次実行
"""

import yaml
import subprocess
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any


class TaskRunner:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"エラー: 設定ファイル '{self.config_path}' が見つかりません")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"エラー: YAML設定ファイルの解析に失敗しました: {e}")
            sys.exit(1)
    
    def _build_command(self, experiment: Dict[str, Any]) -> List[str]:
        """実験設定からコマンドを構築"""
        base_config = self.config.get('base_config', {})
        python_command = base_config.get('python_command','labpython')
        
        # 基本コマンド
        cmd = [
            python_command, 'train.py',
            '--task', base_config.get('task', 'Humanoidprac-v0'),
            '--num_envs',str(base_config.get('num_envs', 1024))
        ]
        
        # ヘッドレスモード
        if base_config.get('headless', True):
            cmd.append('--headless')
        
        # エージェント設定
        directory = base_config.get('directory', 'h1_flat/ex')
        cmd.append(f'agent.agent.experiment.directory="{directory}"')
        
        # トルク設定
        joint_torque = experiment.get('joint_torque', [])
        torque_str = str(joint_torque).replace(' ', '')  # スペースを除去
        cmd.append(f"env.events.change_joint_torque.params.joint_torque={torque_str}")
        
        # ジョイント名設定
        joint_names = experiment.get('joint_names', [])
        names_str = str(joint_names).replace(' ', '')  # スペースを除去
        names_str = names_str.replace("'", '"')
        cmd.append(f"'env.events.change_joint_torque.params.asset_cfg.joint_names={names_str}'")
        # ↑なんかこれシングルクオートで囲って、]はエスケープしない。文字列は""で囲むでいけた。よくわからん。
        
        return cmd
    
    def run_experiment(self, experiment: Dict[str, Any], experiment_num: int) -> bool:
        """単一の実験を実行"""
        print(f"\n{'='*60}")
        print(f"実験 {experiment_num} を開始:")
        print(f"  Joint Torque: {experiment['joint_torque']}")
        print(f"  Joint Names: {experiment['joint_names']}")
        print(f"{'='*60}")
        
        cmd = self._build_command(experiment)
        bash_cmd = f"source ~/.bashrc && {' '.join(cmd)}"

        # コマンドを表示
        print(f"実行コマンド: {bash_cmd}")

        print()
        
        try:
            # コマンド実行
            result = subprocess.run(
                bash_cmd,
                shell=True,
                check=True,
                executable='/bin/bash',
                text=True
            )
            
            print(f"\n実験 {experiment_num} が正常に完了しました")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\nエラー: 実験 {experiment_num} が失敗しました")
            print(f"終了コード: {e.returncode}")
            return False
        except KeyboardInterrupt:
            print(f"\n実験 {experiment_num} が中断されました")
            return False
    
    def run_all_experiments(self, start_from: int = 1, delay: int = 0) -> None:
        """すべての実験を順次実行"""
        experiments = self.config.get('experiments', [])
        
        if not experiments:
            print("エラー: 実験設定が見つかりません")
            return
        
        print(f"設定ファイル: {self.config_path}")
        print(f"実験数: {len(experiments)}")
        print(f"開始実験番号: {start_from}")
        
        if delay > 0:
            print(f"実験間の待機時間: {delay}秒")
        
        successful = 0
        failed = 0
        
        for i, experiment in enumerate(experiments[start_from-1:], start=start_from):
            success = self.run_experiment(experiment, i)
            
            if success:
                successful += 1
            else:
                failed += 1
                response = input(f"\n実験 {i} が失敗しました。続行しますか？ (y/n): ")
                if response.lower() != 'y':
                    break
            
            # 最後の実験でなければ待機
            if i < len(experiments) and delay > 0:
                print(f"\n{delay}秒待機中...")
                time.sleep(delay)
        
        # 結果サマリー
        print(f"\n{'='*60}")
        print("実行結果サマリー:")
        print(f"  成功: {successful}件")
        print(f"  失敗: {failed}件")
        print(f"  合計: {successful + failed}件")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="YAML設定ファイルから複数の実験を順次実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python task_runner.py config.yaml
  python task_runner.py config.yaml --start-from 3
  python task_runner.py config.yaml --delay 30
        """
    )
    
    parser.add_argument(
        'config_file',
        type=str,
        default='task_runner_config.yaml',
        help='YAML設定ファイルのパス'
    )
    
    parser.add_argument(
        '--start-from',
        type=int,
        default=1,
        help='開始する実験番号 (デフォルト: 1)'
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=5,
        help='実験間の待機時間（秒）(デフォルト: 5)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実際には実行せず、コマンドのみ表示'
    )
    
    args = parser.parse_args()
    
    # タスクランナーを初期化
    runner = TaskRunner(args.config_file)
    
    if args.dry_run:
        # ドライランモード: コマンドのみ表示
        experiments = runner.config.get('experiments', [])
        print(f"ドライランモード: {len(experiments)}件の実験")
        print(f"{'='*60}")
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n実験 {i}:")
            print(f"  Joint Torque: {experiment['joint_torque']}")
            print(f"  Joint Names: {experiment['joint_names']}")
            cmd = runner._build_command(experiment)
            print(f"  コマンド: {' '.join(cmd)}")
    else:
        # 実際に実行
        runner.run_all_experiments(
            start_from=args.start_from,
            delay=args.delay
        )


if __name__ == '__main__':
    main()