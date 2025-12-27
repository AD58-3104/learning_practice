import data
import torch
import setting
import math

def round_len(length, base=50):
    return int(base * math.ceil(float(length)/base))

def check_length(datasets):
    length_map = {}
    total_length = 0
    for episode in datasets:
        ep_len = data.get_episode_length(episode)
        total_length += ep_len
        ep_len = round_len(ep_len, base=50)
        if ep_len not in length_map:
            length_map[ep_len] = 0
        length_map[ep_len] += 1
    print("----------<check_length>----------")
    print("Episode Length Distribution:")
    for length in sorted(length_map.keys()):
        print(f"Length {length}: {length_map[length]} episodes")
    print(f"Total length (rounded): {total_length}")

def check_failures(datasets):
    joint_failure_counts = torch.zeros(setting.WHOLE_JOINT_NUM)
    for episode in datasets:
        label = episode["label"]
        label_sum = label.sum(dim=0)
        joint_failure_counts += label_sum
    print("----------<check_failures>----------")
    print(f"Episode Label Sum: {joint_failure_counts}")

def check_std(datasets):
    mu = torch.zeros(setting.OBS_DIMENSION)
    sigma = torch.zeros(setting.OBS_DIMENSION)
    total_length = 0
    for episode in datasets:
        obs = episode["data"]
        mu += obs.sum(dim=0)
        sigma += (obs ** 2).sum(dim=0)
        total_length += data.get_episode_length(episode)
    
    data_std = torch.sqrt(sigma / total_length - (mu / total_length) ** 2)
    torch.set_printoptions(sci_mode=False)
    print("----------<check_std>----------")
    print(f"Data Standard Deviation: {data_std}")
    print(f"Min STD: {data_std.min().item()}, Max STD: {data_std.max().item()}")
    print(f"Mu: {mu / total_length}")
    print(f"Sigma: {sigma / total_length}")

def check_mem_size(datasets):
    import sys
    total_size = 0
    for episode in datasets:
        total_size += sys.getsizeof(episode["data"])
        total_size += sys.getsizeof(episode["label"])
    print("----------<check_mem_size>----------")
    print(f"Total dataset size in memory: {total_size / (1024 ** 2):.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="processed_data", help="Directory containing processed data")
    args = parser.parse_args()
    datasets = data.JointDataset(
                data_dir=args.dir,
                sequence_length=setting.SEQUENCE_LENGTH,
                cache_in_memory=True
                )
    dataloader = torch.utils.data.DataLoader(
                        datasets,
                        batch_size=1,
                        shuffle=True,
                        # collate_fn=data.collate_episodes,
                        num_workers=6,
                        pin_memory=True  # CPU→GPU転送を高速化
                    )
    check_mem_size(datasets)
    check_std(datasets)
    check_length(datasets)
    check_failures(datasets)