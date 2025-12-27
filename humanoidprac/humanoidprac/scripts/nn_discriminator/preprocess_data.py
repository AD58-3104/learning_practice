
import data

datasets = data.JointDataset(
        data_dir="processed_data",
        sequence_length=10,
        cache_in_memory=True
        )
datasets.preprocess_all_data()