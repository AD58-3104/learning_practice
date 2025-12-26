import torch
import data
from data import get_sequence_from_episode, get_episode_length
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import setting

class Trainer:
    def __init__(self, model, active_joint_indices=None):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    # 【重要】マスク処理をするため、ここでは平均(mean)を取らず、個別の損失を返すように設定
        self.error_function = torch.nn.BCELoss(reduction='none')        
        self.step = 0
        self.running_loss = 0.0
        current_datetime = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"learning_log/{current_datetime}")
        
        if active_joint_indices is None:
            # 指定無しなら全ての関節を対象とする
            active_joint_indices = list(range(setting.WHOLE_JOINT_NUM)) # 0~18
        self.joint_mask = torch.zeros((1, 1, setting.WHOLE_JOINT_NUM)).to("cuda")
        for idx in active_joint_indices:
            self.joint_mask[0, 0, idx] = 1.0
        print(f" Joint Mask: {self.joint_mask}")

    def train_step(self, inputs, targets, mask, hidden_states):
        """
        inputs: (Batch, Seq_Len, Input_Dim)
        targets: (Batch, Seq_Len, Output_Dim)
        mask: (Batch, Seq_Len, 1) 
        """
        self.model.train()
        self.optimizer.zero_grad()
        # モデルに入力
        outputs, next_hidden_states = self.model(inputs, hidden=hidden_states)
        # 【修正点】マスクをターゲットと同じ形 (Batch, Seq, 19) に拡張します
        # これにより、sum()を取った時に 19関節分 × タイムステップ の総数が分母になります
        expanded_mask = mask.expand_as(targets)
        # expanded_mask (Batch, Seq, 1) * joint_mask (1, 1, 19) 
        # = final_mask (Batch, Seq, 19)
        final_mask = expanded_mask * self.joint_mask

        # --- 損失の計算 ---
        loss_raw = self.error_function(outputs, targets)
        masked_loss = loss_raw * final_mask
        # 修正: 分母を final_mask.sum() に変更
        loss = masked_loss.sum() / final_mask.sum()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.running_loss += loss.item()
        # --- 精度の計算 ---
        outputs_bin = (outputs > 0.5).float()
        correct = (outputs_bin == targets).float() * final_mask
        # 修正: 分母を final_mask.sum() に変更
        accuracy = correct.sum() / final_mask.sum()
        print_step = 1000
        if self.step % print_step == 0:
            if self.step % 5000 == 0:
                print(f"Step {self.step}, Loss: {self.running_loss / print_step:.4f}")
            self.writer.add_scalar("info / running_loss", self.running_loss / print_step, global_step=self.step)
            self.running_loss = 0.0
        next_hidden_states = next_hidden_states.detach()
        return loss.item(), accuracy.item(), next_hidden_states

    def train_episode(self, episode, sequence_length=100):
        """
        sequence_length: 100などの大きな値（チャンクサイズ）
        """
        ep_len = get_episode_length(episode)
        hidden_states = None
        
        total_loss = 0.0
        total_accuracy = 0.0
        count = 0
        
        # sequence_length ずつ進む (重複なし)
        for idx in range(0, ep_len, sequence_length):
            # データの取得
            # ここでは get_sequence_from_episode が指定範囲 (idx ~ idx+seq_len) を返す関数と仮定
            # 端数（最後の部分）は短いまま返ってくると想定します
            # 返り値shape例: (Batch, Actual_Len, Dim)
            inputs, targets = get_sequence_from_episode(episode, idx, sequence_length)
            
            inputs = inputs.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
            
            # --- パディング処理 ---
            batch_size, current_len, input_dim = inputs.shape
            _, _, output_dim = targets.shape
            
            # 足りない長さを計算
            pad_len = sequence_length - current_len
            
            if pad_len > 0:
                # Inputsのパディング: 後ろに0を追加
                # F.padの引数は (後ろ, 前, 上, 下...) の順で指定（次元の逆順）
                # (Batch, Seq, Dim) の場合、Dim方向はパディングなし(0,0)、Seq方向は(0, pad_len)
                inputs_padded = F.pad(inputs, (0, 0, 0, pad_len), "constant", 0)
                targets_padded = F.pad(targets, (0, 0, 0, pad_len), "constant", 0)
                # マスクの作成: 有効部分(current_len)は1、パディング部分(pad_len)は0
                # shape: (Batch, Seq_Len, 1) ※targetsに合わせてブロードキャスト可能にする
                mask = torch.cat([
                    torch.ones((batch_size, current_len, 1), device="cuda"),
                    torch.zeros((batch_size, pad_len, 1), device="cuda")
                ], dim=1)
            else:
                # パディング不要の場合
                inputs_padded = inputs
                targets_padded = targets
                mask = torch.ones((batch_size, current_len, 1), device="cuda")
            # --- 学習ステップ ---
            loss, accuracy, hidden_states = self.train_step(
                inputs_padded, 
                targets_padded, 
                mask, 
                hidden_states
            )
            total_loss += loss
            total_accuracy += accuracy
            count += 1
        if count == 0:
            return 0.0, 0.0
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

    failure_joint_list = [0, 1, 3, 4, 7, 8, 11, 12]

    model = JointGRUNet(input_size, hidden_size, output_size).to("cuda")
    trainer = Trainer(model, active_joint_indices=failure_joint_list)
    total_loss = 0.0

    print("Starting training... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(args.epoch):  # 10 epochs for demonstration
        start = time.time()
        total_loss = 0.0
        total_accuracy = 0.0
        total_episodes = 1
        for episode in dataloader:
            loss, accuracy = trainer.train_episode(episode=episode, sequence_length=200)
            total_loss += loss
            total_accuracy += accuracy
            trainer.writer.add_scalar("info / loss", total_loss / total_episodes, total_episodes)
            trainer.writer.add_scalar("info / accuracy", total_accuracy / total_episodes, total_episodes)
            total_episodes += 1
        end = time.time()
        trainer.writer.add_scalar("epoch_time", end - start, epoch)
        print(f"Epoch {epoch + 1} completed in {end - start:.2f} seconds")
        trainer.save_model(f"models/joint_net_epoch_{epoch + 1}.pth")
    print(f"Training loss: {total_loss / total_episodes}")
