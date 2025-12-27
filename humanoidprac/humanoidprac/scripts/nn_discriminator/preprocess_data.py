
import data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="processed_data", help="Directory containing processed data")
args = parser.parse_args()

datasets = data.JointDataset(
        data_dir=args.dir,
        sequence_length=10,
        cache_in_memory=True
        )
datasets.preprocess_all_data()