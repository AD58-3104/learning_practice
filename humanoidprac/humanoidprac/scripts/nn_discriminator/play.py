import torch
import data
import time


if __name__ == "__main__":
    from model import JointNet

    input_size = 69    # 観測は69次元
    hidden_size = 64
    output_size = 19   # 19個の関節それぞれに故障があるかどうかを判断

    model = JointNet(input_size, hidden_size, output_size).to("cuda")
    model.load_state_dict(torch.load("models/joint_net_epoch_10.pth"))
    model.eval()

    datasets = data.JointDataset(data_dir="test_data/processed_data", device="cuda")
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=32, shuffle=False, collate_fn=data.collate_episodes)

    total_correct = 0
    total_samples = 0

    print("Starting evaluation... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = total_correct / total_samples
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")