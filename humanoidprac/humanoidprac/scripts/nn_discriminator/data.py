from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from pathlib import Path
import tqdm
from multiprocessing import Pool, cpu_count, Manager, Lock
import bisect
import numpy as np
import copy
from concurrent.futures import ThreadPoolExecutor
import os

min_save_episode_length = 90 # これより短いエピソードは保存しない

def _load_file_pair(args):
    """並列読み込み用のヘルパー関数"""
    key, data_file, label_file, cache_in_memory = args
    try:
        label = np.load(label_file)["data"]
        data = np.load(data_file)["data"]

        label_length = len(label)
        data_length = len(data)

        # 長さの検証
        if label_length != data_length:
            return None, f"Warning: Length mismatch - skipping {data_file.name} ({data_length} rows) and {label_file.name} ({label_length} rows)"

        if label_length == 0:
            return None, None  # 空のファイルはスキップ

        # メモリキャッシュを使用する場合
        if cache_in_memory:
            data_arr = data.astype(np.float32)
            label_arr = label.astype(np.float32)
        else:
            data_arr = None
            label_arr = None

        return {
            'data_file': data_file,
            'label_file': label_file,
            'length': label_length,
            'data_cache': data_arr,
            'label_cache': label_arr
        }, None

    except Exception as e:
        return None, f"Warning: Error loading {data_file.name} or {label_file.name}: {e}"

class DataMerger:
    def __init__(self, output_dir="processed_data", save_count=None, lock=None):
        if save_count is None:
            self.save_count = 0
            self.shared_counter = False
        else:
            self.save_count = save_count
            self.shared_counter = True

        self.lock = lock
        self.output_dir = output_dir + "/"
        self.file_index = 0

    def process(self, obs_data_path, event_data_path, output_file_name):
        # print(f"Processing files: {obs_data_path}, {event_data_path}")

        # データの読み込み
        obs_data_fd = open(obs_data_path, 'rb')
        event_data_fd = open(event_data_path, 'rb')
        obs_data = np.load(obs_data_fd)["data"]
        event_data = np.load(event_data_fd)["data"]

        if obs_data.shape[0] != event_data.shape[0]:
            raise ValueError(f"Observation data and event data length mismatch: {obs_data.shape[0]} vs {event_data.shape[0]}")

        # エピソードの切れ目のインデックスを取得
        terminated_indices = np.where(event_data[:, 1] > 0.5)[0]
        last_index = obs_data.shape[0] - 1

        start_index = 0
        for terminated_index in terminated_indices:
            episode_obs = obs_data[start_index:terminated_index + 1]
            episode_events = event_data[start_index:terminated_index + 1]
            # 先頭2列のstepとterminatedを除いて保存
            self.save_episode(episode_obs[:,2:], episode_events[:,2:], output_file_name)
            start_index = terminated_index + 1

        # 残ったやつを保存
        if start_index <= last_index and (last_index - start_index ) > min_save_episode_length:
            episode_obs = obs_data[start_index:last_index + 1]
            episode_events = event_data[start_index:last_index + 1]
            # 先頭2列のstepとterminatedを除いて保存
            self.save_episode(episode_obs[:,2:], episode_events[:,2:], output_file_name)

    def save_episode(self, obs, events, output_file_name):
        # ミューテックスで保護して save_count を取得・更新
        if self.shared_counter and self.lock is not None:
            with self.lock:
                save_count = self.save_count.value
                self.save_count.value += 1
        else:
            save_count = copy.copy(self.save_count)
            self.save_count += 1

        output_labels_savefile_name = f"{save_count}_{output_file_name}_labels.npz"
        output_observations_savefile_name = f"{save_count}_{output_file_name}_data.npz"

        # ファイルパスを作成
        data_file_name = self.output_dir + output_observations_savefile_name
        label_file_name = self.output_dir + output_labels_savefile_name

        # CSVファイルとして保存（新規作成）
        np.savez_compressed(data_file_name, data=obs)
        np.savez_compressed(label_file_name, data=events)

        # print(f"Saved {obs.shape[0]} lines ➡ {data_file_name} and {events.shape[0]} lines ➡ {label_file_name}")


class JointDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 1,
                 device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 cache_in_memory: bool = True, use_mmap: bool = True,
                 episode_mode: bool = True):
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
        self.episode_mode = episode_mode

        all_data_files = sorted(list(self.data_dir.glob("*_data.npz")))
        all_label_files = sorted(list(self.data_dir.glob("*_labels.npz")))

        # データファイルとラベルファイルのペアを正しく作成
        # ファイル名から _data.npz または _labels.npz を除いた部分をキーとして使用
        data_dict = {}
        for data_file in all_data_files:
            key = str(data_file).replace("_data.npz", "")
            data_dict[key] = data_file

        label_dict = {}
        for label_file in all_label_files:
            key = str(label_file).replace("_labels.npz", "")
            label_dict[key] = label_file

        # 両方に存在するキーのみを使用
        common_keys = sorted(set(data_dict.keys()) & set(label_dict.keys()))

        # 空でないファイルのみをフィルタリング
        self.data_file_list = []
        self.label_file_list = []
        self.episode_lengths = []
        self.cumulative_lengths = [0] # 各エピソード長の累積和（先頭は0）
        self.total_length = 0
        self.total_episodes = 0

        # メモリキャッシュ用
        self.data_cache = []
        self.label_cache = []

        # 並列読み込み処理
        max_workers = min(os.cpu_count() or 4, len(common_keys))
        load_args = [
            (key, data_dict[key], label_dict[key], cache_in_memory)
            for key in common_keys
        ]

        print(f"Loading {len(common_keys)} episodes using {max_workers} threads...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm.tqdm(
                executor.map(_load_file_pair, load_args),
                total=len(load_args),
                desc="Loading episodes"
            ))

        # 結果を集約
        for result, error_msg in results:
            if error_msg:
                print(error_msg)
                continue

            if result is None:
                continue

            self.data_file_list.append(result['data_file'])
            self.label_file_list.append(result['label_file'])
            self.episode_lengths.append(result['length'])
            self.total_length += result['length']
            self.total_episodes += 1
            self.cumulative_lengths.append(self.total_length)
            self.data_cache.append(result['data_cache'])
            self.label_cache.append(result['label_cache'])

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
        if self.episode_mode:
            return self.total_episodes
        else:
            return self.sequence_total_length
        
    def __getitem__(self, idx):
        if self.episode_mode:
            return self.get_episode(idx)
        else:
            return self.get_single_step(idx)

    def get_episode(self, episode_idx):
        if episode_idx < 0:
            episode_idx += self.total_episodes
        if episode_idx < 0 or episode_idx >= self.total_episodes:
            raise IndexError("Index out of range")

        # メモリキャッシュを使用している場合
        if self.cache_in_memory:
            data_arr = self.data_cache[episode_idx]
            label_arr = self.label_cache[episode_idx]
        # メモリマップドファイルを使用する場合
        elif self.use_mmap:
            # npzファイルをメモリマップモードで読み込み
            with np.load(self.data_file_list[episode_idx], mmap_mode='r') as data_file:
                data_arr = data_file['data'].astype(np.float32)
            with np.load(self.label_file_list[episode_idx], mmap_mode='r') as label_file:
                label_arr = label_file['data'].astype(np.float32)
        # npzファイルを通常読み込み
        else:
            with np.load(self.data_file_list[episode_idx]) as data_file:
                data_arr = data_file['data'].astype(np.float32)
            with np.load(self.label_file_list[episode_idx]) as label_file:
                label_arr = label_file['data'].astype(np.float32)

        # テンソル化（全データを含む2次元テンソル）
        data = torch.from_numpy(data_arr).to(dtype=torch.float32)
        label = torch.from_numpy(label_arr).to(dtype=torch.float32)

        return {'data': data, 'label': label}

    def get_single_step(self, idx):
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
            # npzファイルをメモリマップモードで読み込み
            with np.load(self.data_file_list[file_idx], mmap_mode='r') as data_file:
                data_arr = data_file['data'][local_idx:local_idx + self.sequence_length].astype(np.float32)
            with np.load(self.label_file_list[file_idx], mmap_mode='r') as label_file:
                label_arr = label_file['data'][local_idx:local_idx + self.sequence_length].astype(np.float32)
        # npzファイルを通常読み込み
        else:
            with np.load(self.data_file_list[file_idx]) as data_file:
                data_arr = data_file['data'][local_idx:local_idx + self.sequence_length].astype(np.float32)
            with np.load(self.label_file_list[file_idx]) as label_file:
                label_arr = label_file['data'][local_idx:local_idx + self.sequence_length].astype(np.float32)

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

def collate_singlesteps(batch):
    data = torch.stack([b['data'] for b in batch], dim=0)
    label = torch.stack([b['label'] for b in batch], dim=0)
    return data, label

def collate_episodes(batch):
    max_len = max(ep['data'].shape[0] for ep in batch)
    padded_data = []
    padded_label = []
    for ep in batch:
        pad_len = max_len - ep['data'].shape[0]
        padded_data.append(F.pad(ep['data'], (0, 0, 0, pad_len)))
        padded_label.append(F.pad(ep['label'], (0, 0, 0, pad_len)))
    return torch.stack(padded_data), torch.stack(padded_label)

def get_sequence_from_episode(episode, index, seq_length):
    """
    エピソードから指定されたインデックスとシーケンス長でデータを取得
    指定されたエピソード長に満たない場合、その長さまでを取得する
    """
    data = episode['data']
    label = episode['label']
    if index + seq_length > data.shape[-2]:
        seq_length = data.shape[-2] - index
    # バッチ次元を保持しつつ、時間次元をスライス
    data = data[:, index:index + seq_length, :]
    label = label[:, index:index + seq_length, :]
    return data, label

def get_episode_length(episode):
    # shape[1]が時間次元（バッチ次元の次）
    return episode['data'].shape[-2]
 
def file_counts_in_directory(directory: str):
    import os
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(files)

def process_single_file(args):
    """並列処理用の関数：単一ファイルを処理する"""
    i, output_dir, save_count, lock = args
    merger = DataMerger(output_dir=output_dir, save_count=save_count, lock=lock)
    merger.process(
        obs_data_path=f"nn_data/discriminator_obs_env_{i}.npz",
        event_data_path=f"nn_data/joint_torque_env_{i}.npz",
        output_file_name=f"nn_discriminator_training_data"
    )

    return i

if __name__ == "__main__":
    num_target_files = int(file_counts_in_directory("nn_data") / 2)
    output_dir = "processed_data"

    # 並列処理の実行
    num_workers = min(cpu_count(), num_target_files)
    print(f"Processing {num_target_files} files using {num_workers} workers...")

    # マネージャーを使って共有カウンターとロックを作成
    with Manager() as manager:
        save_count = manager.Value('i', 0)  # 共有整数カウンター
        lock = manager.Lock()  # プロセス間で共有されるロック

        with Pool(processes=num_workers) as pool:
            args_list = [(i, output_dir, save_count, lock) for i in range(num_target_files)]
            # tqdmでプログレスバーを表示
            results = list(tqdm.tqdm(
                pool.imap(process_single_file, args_list),
                total=num_target_files,
                desc="Processing files"
            ))

    print("All files processed successfully!")

    dataset = JointDataset(data_dir="processed_data")
    print(f"Dataset length: {len(dataset)}")