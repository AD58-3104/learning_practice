import data
import setting

# Create dataset
dataset = data.JointDataset(
    data_dir="processed_data",
    sequence_length=setting.SEQUENCE_LENGTH,
    cache_in_memory=False,
    use_mmap=True
)

print(f"Total sequences: {len(dataset)}")
print(f"Total files: {dataset.file_num}")
print(f"Sequence length: {dataset.sequence_length}")

# Find which file contains index 1478361
target_idx = 1478361
import bisect
file_idx = bisect.bisect_right(dataset.sequence_cumulative_lengths, target_idx) - 1
local_idx = target_idx - dataset.sequence_cumulative_lengths[file_idx]

print(f"\nIndex {target_idx}:")
print(f"  File index: {file_idx}")
print(f"  Local index: {local_idx}")
print(f"  Episode length: {dataset.episode_lengths[file_idx]}")
print(f"  Data file: {dataset.data_file_list[file_idx]}")
print(f"  Label file: {dataset.label_file_list[file_idx]}")
print(f"  Num sequences in this file: {max(0, dataset.episode_lengths[file_idx] - dataset.sequence_length + 1)}")
print(f"  Cumulative lengths around this file: {dataset.sequence_cumulative_lengths[file_idx:file_idx+3]}")

# Check file integrity
import pandas as pd
try:
    data_df = pd.read_csv(dataset.data_file_list[file_idx], header=None)
    label_df = pd.read_csv(dataset.label_file_list[file_idx], header=None)
    print(f"  Actual data rows: {len(data_df)}")
    print(f"  Actual label rows: {len(label_df)}")
except Exception as e:
    print(f"  Error reading file: {e}")
