import torch
import data
from data import get_sequence_from_episode, get_episode_length
import time
from torch.utils.tensorboard import SummaryWriter
import setting

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.error_function = torch.nn.BCELoss()
        self.step = 0
        self.running_loss = 0.0
        current_datetime = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"learning_log/{current_datetime}")

    def train_step(self, inputs, targets, hidden_states=None):
        self.model.train()
        self.optimizer.zero_grad()
        outputs, hidden_states = self.model(inputs, hidden=hidden_states)
        loss = self.error_function(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.running_loss += loss.item()
        outputs = (outputs > 0.5).long()
        targets = targets.long()
        accuracy = (outputs == targets).sum().item() / targets.numel()
        if self.step % 5000 == 0:
            print(f"Step {self.step}, Loss: {self.running_loss / 5000}")
            self.writer.add_scalar("loss", self.running_loss / 5000, global_step=self.step)
            self.running_loss = 0.0
        # 計算グラフから切り離して次のステップで再利用可能にする
        return loss.item(), accuracy, hidden_states.detach()

    def train_episode(self, episode, sequence_length):
        ep_len = get_episode_length(episode)
        hidden_states = None
        total_loss = 0.0
        total_accuracy = 0.0
        count = 0
        for idx in range(0, ep_len - sequence_length + 1):
            inputs, targets = get_sequence_from_episode(episode, idx, sequence_length)
            inputs = inputs.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
            loss, accuracy, hidden_states = self.train_step(inputs, targets[:, -1, :], hidden_states)
            total_loss += loss
            total_accuracy += accuracy
            count += 1
        return total_loss / count, total_accuracy / count


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    from joint_model import JointGRUNet
    import argparse

    parser = argparse.ArgumentParser(description="Train a joint network model")
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    sequence_length = setting.SEQUENCE_LENGTH
    input_size = setting.OBS_DIMENSION    # 観測は88次元
    hidden_size = setting.HIDDEN_SIZE
    output_size = setting.WHOLE_JOINT_NUM   # 19個の関節それぞれに故障があるかどうかを判断

    datasets = data.JointDataset(
                data_dir="processed_data",
                sequence_length=sequence_length,
                cache_in_memory=True
                )
    dataloader = torch.utils.data.DataLoader(
                        datasets,
                        batch_size=1,
                        shuffle=True,
                        # collate_fn=data.collate_episodes,
                        num_workers=6,
                        pin_memory=True  # CPU→GPU転送を高速化
                    )

    model = JointGRUNet(input_size, hidden_size, output_size).to("cuda")
    trainer = Trainer(model)
    total_loss = 0.0

    print("Starting training... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(args.epoch):  # 10 epochs for demonstration
        start = time.time()
        total_loss = 0.0
        total_accuracy = 0.0
        total_episodes = 1
        for episode in dataloader:
            loss, accuracy = trainer.train_episode(episode=episode, sequence_length=sequence_length)
            total_loss += loss
            total_accuracy += accuracy
            trainer.writer.add_scalar("loss", total_loss / total_episodes, total_episodes)
            trainer.writer.add_scalar("accuracy", total_accuracy / total_episodes, total_episodes)
            total_episodes += 1
        end = time.time()
        trainer.writer.add_scalar("epoch_time", end - start, epoch)
        print(f"Epoch {epoch + 1} completed in {end - start:.2f} seconds")
        trainer.save_model(f"models/joint_net_epoch_{epoch + 1}.pth")
    print(f"Training loss: {total_loss / total_episodes}")
