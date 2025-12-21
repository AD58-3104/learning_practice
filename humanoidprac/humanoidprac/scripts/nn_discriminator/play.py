import torch
import data
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

    model = JointGRUNet(input_size, hidden_size, output_size).to("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    batch_size = 256
    datasets = data.JointDataset(data_dir="test_data/processed_data", device="cuda",sequence_length=setting.SEQUENCE_LENGTH)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=False, collate_fn=data.collate_episodes,num_workers=4)
    total_samples = 0
    batch_index = 0
    hidden_states = None
    total_correct = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")
    total_real_failure = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")
    print("Starting evaluation... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def print_result():
        for joint_id in range(setting.WHOLE_JOINT_NUM):
            joint_accuracy = total_correct[joint_id].item() / total_samples
            print(f"Joint {joint_id} accuracy: {joint_accuracy * 100:.2f}% , Real failures: {total_real_failure[joint_id].item()} samples")

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            inputs, targets = batch
            if inputs.size(0) != batch_size:
                break # バッチサイズがbatch_sizeでない場合は終了
            outputs, hidden_states = model(inputs, hidden_states)
            outputs = (outputs > 0.6).long()
            targets = targets[:,-1,:].long()
            total_real_failure += targets.sum(dim=0)
            total_correct += (outputs == targets).long().sum(dim=0)
            total_samples += targets.size(0)
            batch_index += 1
            if batch_index % 300 == 0:
                print("Current accuracy after {batch_index} batches")
                print_result()
    print("Final evaluation results:")
    print_result()