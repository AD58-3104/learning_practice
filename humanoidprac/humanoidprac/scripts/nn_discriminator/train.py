import torch
import data
import time
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.error_function = torch.nn.BCEWithLogitsLoss()
        self.step = 0
        self.running_loss = 0.0
        self.writer = SummaryWriter(log_dir="learning_log")

    def train_step(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.error_function(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.running_loss += loss.item()
        outputs = (outputs > 0.6).long()
        targets = targets.long()
        accuracy = (outputs == targets).sum().item() / targets.numel()
        if self.step % 1000 == 0:
            print(f"Step {self.step}, Loss: {self.running_loss / 1000}")
            # tf.summary.scalar("loss", self.running_loss / 1000, step=self.step)
            self.running_loss = 0.0
        return loss.item(), accuracy
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    from model import JointNet, JointGRUNet
    import argparse

    parser = argparse.ArgumentParser(description="Train a joint network model")
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    sequence_length = 10
    input_size = 69    # 観測は69次元
    hidden_size = 128
    output_size = 19   # 19個の関節それぞれに故障があるかどうかを判断

    datasets = data.JointDataset(data_dir="processed_data",sequence_length=sequence_length,device="cuda")
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=32,shuffle=True,collate_fn=data.collate_episodes)

    model = JointGRUNet(input_size, hidden_size, output_size).to("cuda")
    trainer = Trainer(model)
    total_loss = 0.0

    print("Starting training... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(args.epoch):  # 10 epochs for demonstration
        start = time.time()
        total_loss = 0.0
        total_accuracy = 0.0
        for batch in dataloader:
            inputs, targets = batch
            loss, accuracy = trainer.train_step(inputs, targets[:,-1,:])
            total_loss += loss
            total_accuracy += accuracy
        end = time.time()
        trainer.writer.add_scalar("loss", total_loss / len(dataloader), epoch)
        trainer.writer.add_scalar("epoch_time", end - start, epoch)
        trainer.writer.add_scalar("accuracy", total_accuracy / len(dataloader), epoch)
        print(f"Epoch {epoch + 1} completed in {end - start:.2f} seconds")
        trainer.save_model(f"models/joint_net_epoch_{epoch + 1}.pth")
    print(f"Training loss: {total_loss / len(dataloader)}")
