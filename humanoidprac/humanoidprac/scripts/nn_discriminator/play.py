import torch
import data
import time


if __name__ == "__main__":
    from model import JointGRUNet

    input_size = 69    # 観測は69次元
    hidden_size = 128
    output_size = 19   # 19個の関節それぞれに故障があるかどうかを判断

    model = JointGRUNet(input_size, hidden_size, output_size).to("cuda")
    model.load_state_dict(torch.load("models/joint_net_epoch_5.pth"))
    model.eval()

    datasets = data.JointDataset(data_dir="test_data/processed_data", device="cuda",sequence_length=10)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=32, shuffle=False, collate_fn=data.collate_episodes)

    total_correct = 0
    total_samples = 0

    print("Starting evaluation... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            outputs = (outputs > 0.6).long()
            targets = targets[:,-1,:].long()
            total_correct += (outputs == targets).sum().item()
            total_samples += targets.numel()

    accuracy = total_correct / total_samples
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")
