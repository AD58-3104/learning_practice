import torch
import nn_data
from nn_data import get_sequence_from_episode, get_episode_length
import time
import argparse
import tqdm
import setting
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def get_latest_model_path(model_dir="models")-> str:
    import os
    model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    sorted_dirs = sorted(model_dirs,reverse=True)
    if sorted_dirs:
        latest_dir = sorted_dirs[0]
        models = [f for f in os.listdir(os.path.join(model_dir, latest_dir)) if f.endswith('.pth')]
        sorted_models = sorted(models, key=lambda x: int(x.split('_')[-1].split('.pth')[0]))  # エポック番号でソート
        model_path = os.path.join(model_dir, latest_dir, sorted_models[-1])
        print(f"Using latest model at: {model_path}")
        return model_path
    else:
        raise FileNotFoundError("No model directories found.")

if __name__ == "__main__":
    from joint_model import JointGRUNet

    parser = argparse.ArgumentParser(description="Evaluate JointGRUNet model")
    parser.add_argument("--model-path", type=str, default=get_latest_model_path(), help="Path to the trained model")
    args = parser.parse_args()

    sequence_length = setting.SEQUENCE_LENGTH
    input_size = setting.OBS_DIMENSION    # 観測は88次元
    hidden_size = setting.HIDDEN_SIZE
    output_size = setting.WHOLE_JOINT_NUM   # 19個の関節それぞれに故障があるかどうかを判断
    num_layers = setting.NUM_LAYERS      # GRUの層数
    chunk_size = 1
    max_grad_norm = setting.MAX_GRAD_NORM # 勾配クリッピングの最大ノルム
    batch_size = setting.BATCH_SIZE

    model = JointGRUNet(input_size, hidden_size, output_size,num_layers=num_layers).to("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    batch_size = 1
    datasets = nn_data.JointDataset(
                        data_dir="test_data/processed_data",
                        sequence_length=sequence_length,
                        cache_in_memory=True
                        )
    dataloader = torch.utils.data.DataLoader(
                            datasets, 
                            batch_size=batch_size,
                            shuffle=False,
                            # collate_fn=nn_data.collate_fn_pad_batch,
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
            for ep_idx in range(0, ep_len - chunk_size, chunk_size):
                inputs, targets = get_sequence_from_episode(episode, ep_idx, chunk_size)
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