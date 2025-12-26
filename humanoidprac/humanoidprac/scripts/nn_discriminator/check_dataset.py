import data
import torch
import setting
import math

def round_len(length, base=50):
    return int(base * math.ceil(float(length)/base))

def check_length(datasets):
    length_map = {}
    for episode in datasets:
        ep_len = data.get_episode_length(episode)
        ep_len = round_len(ep_len, base=50)
        if ep_len not in length_map:
            length_map[ep_len] = 0
        length_map[ep_len] += 1
    print("Episode Length Distribution:")
    for length in sorted(length_map.keys()):
        print(f"Length {length}: {length_map[length]} episodes")


if __name__ == "__main__":
    datasets = data.JointDataset(
                data_dir="processed_data",
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
    check_length(datasets)