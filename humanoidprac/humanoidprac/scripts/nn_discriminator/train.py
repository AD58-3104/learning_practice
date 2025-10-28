import torch
import data
import time

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.error_function = torch.nn.CrossEntropyLoss()
        self.step = 0
        self.running_loss = 0.0

    def train_step(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.error_function(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.running_loss += loss.item()
        if self.step % 1000 == 0:
            print(f"Step {self.step}, Loss: {self.running_loss / 1000}")
            # tf.summary.scalar("loss", self.running_loss / 1000, step=self.step)
            self.running_loss = 0.0
        return loss.item()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    from model import JointNet

    input_size = 69    # 観測は69次元
    hidden_size = 64
    output_size = 19   # 19個の関節それぞれに故障があるかどうかを判断

    datasets = data.JointDataset(data_dir="processed_data",device="cuda")
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=32,shuffle=True,collate_fn=data.collate_episodes)

    model = JointNet(input_size, hidden_size, output_size).to("cuda")
    trainer = Trainer(model)
    total_loss = 0.0

    print("Starting training... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(10):  # 10 epochs for demonstration
        start = time.time()
        for batch in dataloader:
            inputs, targets = batch
            loss = trainer.train_step(inputs, targets)
            total_loss += loss
        end = time.time()
        print(f"Epoch {epoch + 1} completed in {end - start:.2f} seconds")
        trainer.save_model(f"models/joint_net_epoch_{epoch + 1}.pth")
    print(f"Training loss: {total_loss / len(dataloader)}")
