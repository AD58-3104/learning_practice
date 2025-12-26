import torch
import data
from data import get_sequence_from_episode, get_episode_length
import time
import argparse
import tqdm
import setting
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

if __name__ == "__main__":
    from joint_model import JointGRUNet

    parser = argparse.ArgumentParser(description="Evaluate JointGRUNet model")
    parser.add_argument("--model-path", type=str, default="models/joint_net_epoch_5.pth", help="Path to the trained model")
    args = parser.parse_args()

    input_size = setting.OBS_DIMENSION    # 観測は88次元
    hidden_size = setting.HIDDEN_SIZE
    output_size = setting.WHOLE_JOINT_NUM   # 19個の関節それぞれに故障があるかどうかを判断
    sequence_length = setting.SEQUENCE_LENGTH

    model = JointGRUNet(input_size, hidden_size, output_size).to("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    batch_size = 1
    datasets = data.JointDataset(
                        data_dir="test_data/processed_data",
                        sequence_length=sequence_length,
                        cache_in_memory=True
                        )
    dataloader = torch.utils.data.DataLoader(
                            datasets, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            # collate_fn=data.collate_episodes,
                            num_workers=4
                        )
    total_samples = 0
    batch_index = 0
    hidden_states = None
    total_correct = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")
    total_real_failure = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")
    total_detect = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")
    print("Starting evaluation... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def print_result():
        for joint_id in range(setting.WHOLE_JOINT_NUM):
            if total_real_failure[joint_id].item() != 0:
                joint_accuracy = total_correct[joint_id].item() / total_real_failure[joint_id].item()
            else:
                joint_accuracy = 0.0
            print(f"Joint {joint_id} accuracy: {joint_accuracy * 100:.2f}% , {total_detect[joint_id].item()}  / {total_real_failure[joint_id].item()} [Detected failures / Real failures]")

    with torch.no_grad():
        for episode in tqdm.tqdm(dataloader, desc="Evaluating"):
            ep_len = get_episode_length(episode)
            hidden_states = None
            for ep_idx in range(0, ep_len - sequence_length, sequence_length):
                inputs, targets = get_sequence_from_episode(episode, ep_idx, sequence_length)
                inputs = inputs.to("cuda", non_blocking=True)
                targets = targets.to("cuda", non_blocking=True)

                # モデルの出力: (batch, seq_len, 19)
                outputs, hidden_states = model(inputs, hidden=hidden_states)
                outputs = (outputs > 0.5).long()
                targets = targets.long()

                # バッチ次元とシーケンス次元をまとめて集計
                # sum(dim=0)でバッチ次元を消し、sum(dim=0)でシーケンス次元を消す
                # または sum(dim=(0,1)) で一度に両方消す
                total_real_failure += targets.sum(dim=(0, 1))  # -> [19]
                total_detect += outputs.sum(dim=(0, 1))  # -> [19]
                total_correct += (outputs & targets).long().sum(dim=(0, 1))  # -> [19]
                total_samples += targets.size(0) * targets.size(1)  # batch * seq_len
            batch_index += 1
            if batch_index % 300 == 0:
                print(f"Current accuracy after {batch_index} batches")
                print_result()
    print("Final evaluation results:")
    print_result()