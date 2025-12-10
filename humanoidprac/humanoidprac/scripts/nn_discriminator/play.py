import torch
import data
import time
import argparse
import tqdm
import setting

if __name__ == "__main__":
    from joint_model import JointGRUNet

    parser = argparse.ArgumentParser(description="Evaluate JointGRUNet model")
    parser.add_argument("--model-path", type=str, default="models/joint_net_epoch_5.pth", help="Path to the trained model")
    args = parser.parse_args()

    input_size = setting.OBS_DIMENSION    # 観測は88次元
    hidden_size = setting.HIDDEN_SIZE
    output_size = setting.WHOLE_JOINT_NUM   # 19個の関節それぞれに故障があるかどうかを判断

    model = JointGRUNet(input_size, hidden_size, output_size).to("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    datasets = data.JointDataset(data_dir="test_data/processed_data", device="cuda",sequence_length=setting.SEQUENCE_LENGTH)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=32, shuffle=False, collate_fn=data.collate_episodes)

    total_correct = 0
    total_samples = 0
    batch_index = 0
    hidden_states = None
    print("Starting evaluation... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            inputs, targets = batch
            if inputs.size(0) != 32:
                break # バッチサイズが32でない場合は終了
            outputs, hidden_states = model(inputs, hidden_states)
            outputs = (outputs > 0.6).long()
            targets = targets[:,-1,:].long()
            total_correct += (outputs == targets).sum().item()
            total_samples += targets.numel()
            batch_index += 1
            if batch_index % 300 == 0:
                print(f"Current accuracy after {batch_index} batches: {total_correct / total_samples * 100:.2f}%")

    accuracy = total_correct / total_samples
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")
