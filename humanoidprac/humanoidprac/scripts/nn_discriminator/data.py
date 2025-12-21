import pandas as pd
from torch.utils.data import Dataset
import torch
import setting
from pathlib import Path
import typing
from typing import TextIO
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import bisect
import numpy as np

class DataMerger:
    def __init__(self, output_dir="processed_data"):
        self.obs_data_fd: typing.Optional[TextIO] = None
        self.event_data_fd: typing.Optional[TextIO] = None
        self.output_labels = []
        self.output_observations = []
        self.save_count = 0
        self.output_dir = output_dir + "/"

    def file_setup(self,obs_data_path, event_data_path, output_file_name):
        if self.obs_data_fd is not None:
            self.obs_data_fd.close()
        if self.event_data_fd is not None:
            self.event_data_fd.close()
        self.obs_data_fd = open(obs_data_path, 'r')
        self.event_data_fd = open(event_data_path, 'r')
        self.output_labels_savefile_name = output_file_name + "_labels.csv"
        self.output_observations_savefile_name = output_file_name + "_data.csv"
        self.output_labels = []
        self.output_observations = []
        print(f"Processing files: {obs_data_path}, {event_data_path}")

    def process(self):
        obs_data = pd.read_csv(self.obs_data_fd)
        event_data = pd.read_csv(self.event_data_fd)

        # terminatedカラムを確実にbool型に変換
        obs_data['terminated'] = obs_data['terminated'].astype(str).str.lower() == 'true'

        # イベントデータをステップカウンターをキーとした辞書に変換
        # 同じステップで複数のイベントが発生する可能性に対応
        event_dict = {}
        for event in event_data.itertuples():
            step = int(event.common_step_counter)
            joint_id = int(event.target_joint_id)
            if step not in event_dict:
                event_dict[step] = []
            event_dict[step].append(joint_id)

        # 初期化用のラベル（全関節0）
        zero_label = [0 for _ in range(setting.WHOLE_JOINT_NUM)]
        label = zero_label.copy()
        for obs in obs_data.itertuples():
            step = int(obs.step)

            # このステップでイベントが発生しているか確認
            if step in event_dict:
                # イベントが発生している関節にラベル1を設定（状態を保持）
                for joint_id in event_dict[step]:
                    label[joint_id] = 1

            # コピーを追加（イベント発生後の状態を保持するため）
            self.output_labels.append(label.copy())
            # itertuples()は [Index, step, terminated, observation_0, ...] の順なので obs[3:] で観測データを取得
            self.output_observations.append(list(obs[3:]))

            # エピソードが終了したら保存
            if obs.terminated:
                self.save_to_files()
                # ラベルをリセットして次のエピソードへ
                label = zero_label.copy()

        self.save_to_files()  # 最後に残ったデータを保存（最後のエピソード）

    def save_to_files(self):
        # データがない場合は保存をスキップ
        if not self.output_observations or not self.output_labels:
            return

        # DataFrameに変換した
        data = pd.DataFrame(self.output_observations)
        label = pd.DataFrame(self.output_labels)

        # ファイルパスを作成
        data_file_name = self.output_dir + str(self.save_count) + '_' + self.output_observations_savefile_name
        label_file_name = self.output_dir + str(self.save_count) + '_' + self.output_labels_savefile_name

        # CSVファイルとして保存（新規作成）
        data.to_csv(data_file_name, index=False, header=False)
        label.to_csv(label_file_name, index=False, header=False)

        print(f"Saved {data.shape[0]} lines ➡ {data_file_name} and {label.shape[0]} lines ➡ {label_file_name}")

        self.save_count += 1
        # リセットする
        self.output_labels = []
        self.output_observations = []


class JointDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 1,
                 device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 cache_in_memory: bool = False, use_mmap: bool = True):
        """
        Args:
            data_dir: データディレクトリのパス
            sequence_length: シーケンス長
            device: Tensorを配置するデバイス
            cache_in_memory: True の場合、全データをメモリにキャッシュ（高速だがメモリ消費大）
            use_mmap: True の場合、メモリマップドファイルを使用（メモリ効率的で高速）
        """
        self.device = device
        self.data_dir = Path(data_dir)
        self.cache_in_memory = cache_in_memory
        self.use_mmap = use_mmap

        all_data_files = sorted(list(self.data_dir.glob("*_data.csv")))
        all_label_files = sorted(list(self.data_dir.glob("*_labels.csv")))

        # データファイルとラベルファイルのペアを正しく作成
        # ファイル名から _data.csv または _labels.csv を除いた部分をキーとして使用
        data_dict = {}
        for data_file in all_data_files:
            key = str(data_file).replace("_data.csv", "")
            data_dict[key] = data_file

        label_dict = {}
        for label_file in all_label_files:
            key = str(label_file).replace("_labels.csv", "")
            label_dict[key] = label_file

        # 両方に存在するキーのみを使用
        common_keys = sorted(set(data_dict.keys()) & set(label_dict.keys()))

        # 空でないファイルのみをフィルタリング
        self.data_file_list = []
        self.label_file_list = []
        self.episode_lengths = []
        self.cumulative_lengths = [0] # 各エピソード長の累積和（先頭は0）
        self.total_length = 0

        # メモリキャッシュ用
        self.data_cache = []
        self.label_cache = []

        for key in common_keys:
            data_file = data_dict[key]
            label_file = label_dict[key]
            try:
                label = pd.read_csv(label_file, header=None)
                data = pd.read_csv(data_file, header=None, dtype=np.float32)

                label_length = len(label)
                data_length = len(data)

                # 長さの検証
                if label_length != data_length:
                    print(f"Warning: Length mismatch - skipping {data_file.name} ({data_length} rows) and {label_file.name} ({label_length} rows)")
                    continue

                if label_length == 0:
                    continue  # 空のファイルはスキップ

                self.data_file_list.append(data_file)
                self.label_file_list.append(label_file)
                self.episode_lengths.append(label_length)
                self.total_length += label_length
                self.cumulative_lengths.append(self.total_length)

                # メモリキャッシュを使用する場合
                if cache_in_memory:
                    data_arr = data.values
                    label_arr = label.values.astype(np.float32)
                    self.data_cache.append(data_arr)
                    self.label_cache.append(label_arr)
                else:
                    self.data_cache.append(None)
                    self.label_cache.append(None)

            except pd.errors.EmptyDataError:
                # 空のファイルはスキップ
                continue

        self.file_num = len(self.data_file_list)
        self.sequence_length = sequence_length  # 時系列ではないやつなら1にする。
        self._update_sequence_cumulative_lengths()

    def _update_sequence_cumulative_lengths(self):
        """sequence_lengthを考慮した累積長を再計算"""
        self.sequence_cumulative_lengths = [0]
        self.sequence_total_length = 0
        for length in self.episode_lengths:
            # 各ファイルから取れるシーケンス数 = max(0, length - sequence_length + 1)
            num_sequences = max(0, length - self.sequence_length + 1)
            self.sequence_total_length += num_sequences
            self.sequence_cumulative_lengths.append(self.sequence_total_length)

    def set_sequence_length(self, length: int):
        self.sequence_length = length
        self._update_sequence_cumulative_lengths()

    def __len__(self):
        return self.sequence_total_length

    def __getitem__(self, idx):
        # 負のインデックス対応
        if idx < 0:
            idx += self.sequence_total_length
        if idx < 0 or idx >= self.sequence_total_length:
            raise IndexError("Index out of range")

        # idxがどのエピソード(ファイル)に属するかを二分探索で求める
        # sequence_cumulative_lengths = [0, seq_num0, seq_num0+seq_num1, ..., total]
        file_idx = bisect.bisect_right(self.sequence_cumulative_lengths, idx) - 1
        local_idx = idx - self.sequence_cumulative_lengths[file_idx]

        # メモリキャッシュを使用している場合
        if self.cache_in_memory:
            data_arr = self.data_cache[file_idx][local_idx:local_idx + self.sequence_length]
            label_arr = self.label_cache[file_idx][local_idx:local_idx + self.sequence_length]
        # メモリマップドファイルを使用する場合
        elif self.use_mmap:
            # numpy.loadtxtよりも高速なnp.genfromtxtを使用し、メモリマップで読む
            data_arr = np.loadtxt(
                self.data_file_list[file_idx],
                delimiter=',',
                skiprows=local_idx,
                max_rows=self.sequence_length,
                dtype=np.float32
            )
            label_arr = np.loadtxt(
                self.label_file_list[file_idx],
                delimiter=',',
                skiprows=local_idx,
                max_rows=self.sequence_length,
                dtype=np.float32
            )
        # pandas を使用する場合（デフォルト）
        else:
            data_rows = pd.read_csv(
                self.data_file_list[file_idx],
                skiprows=local_idx,
                nrows=self.sequence_length,
                header=None,
                dtype=np.float32
            )
            label_rows = pd.read_csv(
                self.label_file_list[file_idx],
                skiprows=local_idx,
                nrows=self.sequence_length,
                header=None,
                dtype=np.float32
            )
            data_arr = data_rows.values
            label_arr = label_rows.values

        # データが期待するsequence_lengthに満たない場合はエラー
        # （__len__で計算済みなので通常は発生しないが、念のため）
        if data_arr.shape[0] != self.sequence_length:
            raise ValueError(f"Expected sequence_length {self.sequence_length}, but got {data_arr.shape[0]} at index {idx} (file_idx={file_idx}, local_idx={local_idx}, episode_length={self.episode_lengths[file_idx]})")

        # テンソル化（sequence_length x feature_dim の2次元テンソル）
        # DataLoaderのマルチプロセスに対応するため、CPUテンソルとして返す
        data = torch.from_numpy(data_arr).to(dtype=torch.float32)
        label = torch.from_numpy(label_arr).to(dtype=torch.float32)

        # sequence_length=1 の場合は従来通り1次元ベクトルとして返す
        if self.sequence_length == 1:
            data = data.squeeze(0)
            label = label.squeeze(0)

        return {'data': data, 'label': label}

def collate_episodes(batch):
    data = torch.stack([b['data'] for b in batch], dim=0)
    label = torch.stack([b['label'] for b in batch], dim=0)
    return data, label


def file_counts_in_directory(directory: str):
    import os
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

def process_single_file(args):
    """並列処理用の関数：単一ファイルを処理する"""
    i, output_dir = args
    merger = DataMerger(output_dir=output_dir)
    merger.file_setup(
        obs_data_path=f"nn_data/discriminator_obs_env_{i}.dat",
        event_data_path=f"nn_data/joint_torque_event_log_env_{i}.dat",
        output_file_name=f"nn_discriminator_training_data_env_{i}"
    )
    merger.process()
    return i

if __name__ == "__main__":
    num_target_files = int(file_counts_in_directory("nn_data") / 2)
    output_dir = "processed_data"

    # 並列処理の実行
    num_workers = min(cpu_count(), num_target_files)
    print(f"Processing {num_target_files} files using {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        args_list = [(i, output_dir) for i in range(num_target_files)]
        # tqdmでプログレスバーを表示
        results = list(tqdm.tqdm(
            pool.imap(process_single_file, args_list),
            total=num_target_files,
            desc="Processing files"
        ))

    print("All files processed successfully!")

    dataset = JointDataset(data_dir="processed_data")
    print(f"Dataset length: {len(dataset)}")