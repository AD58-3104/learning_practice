import torch
import data
import time
import argparse
import tqdm
import setting
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate data statistics")
    parser.add_argument("--target", type=str, default="processed_data", help="Not used in this script")
    args = parser.parse_args()

    input_size = setting.OBS_DIMENSION    # 観測は88次元
    hidden_size = setting.HIDDEN_SIZE
    output_size = setting.WHOLE_JOINT_NUM   # 19個の関節それぞれに故障があるかどうかを判断

    batch_size = 1024
    datasets = data.JointDataset(data_dir=args.target, device="cuda",sequence_length=setting.SEQUENCE_LENGTH)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=False, collate_fn=data.collate_episodes,num_workers=4)
    total_samples = 0
    batch_index = 0
    total_failure_count = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")
    print("Starting evaluation... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def print_result():
        for joint_id in range(setting.WHOLE_JOINT_NUM):
            print(f"Joint {joint_id} failures {total_failure_count[joint_id].item()} samples")
        print(f"Total samples evaluated: {total_samples}")

    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        inputs, targets = batch
        if inputs.size(0) != batch_size:
            break # バッチサイズがbatch_sizeでない場合は終了
        total_failure_count += targets.long().sum(dim=(0,1))
        total_samples += targets.size(0) * targets.size(1)
        batch_index += 1
        if batch_index % 300 == 0:
            print("Current accuracy after {batch_index} batches")
            print_result()
    print("Final evaluation results:")
    print_result()