import torch
import data
from data import get_sequence_from_episode, get_episode_length
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import setting
import yaml
import os

class Trainer:
    def __init__(self, model, active_joint_indices=None, max_grad_norm=1.0, logdir = "learning_log/" + time.strftime("%Y%m%d-%H%M%S")):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # 【重要】マスク処理をするため、ここでは平均(mean)を取らず、個別の損失を返すように設定
        self.error_function = torch.nn.BCELoss(reduction='none')        
        self.step = 0
        self.running_loss = 0.0
        self.writer = SummaryWriter(log_dir=logdir)
        self.max_grad_norm = max_grad_norm
        
        if active_joint_indices is None:
            # 指定無しなら全ての関節を対象とする
            active_joint_indices = list(range(setting.WHOLE_JOINT_NUM)) # 0~18
        self.joint_mask = torch.zeros((1, 1, setting.WHOLE_JOINT_NUM)).to("cuda")
        for idx in active_joint_indices:
            self.joint_mask[0, 0, idx] = 1.0
        print(f" Joint Mask: {self.joint_mask}")
        self.save_path = f"models/{logdir.split('/')[-1]}"
        os.makedirs(self.save_path, exist_ok=True) 


    def train_step(self, inputs, targets, mask, hidden_states):
        self.model.train()
        self.optimizer.zero_grad()
        outputs, next_hidden_states = self.model(inputs, hidden=hidden_states)
        # マスクの適用
        expanded_mask = mask.expand_as(targets)
        final_mask = expanded_mask * self.joint_mask
        # 損失計算 (Batch全体で一括計算)
        loss_raw = self.error_function(outputs, targets)
        masked_loss = loss_raw * final_mask
        # 分母が0になるのを防ぐための小さな値を足すガードを入れるのが一般的です
        loss = masked_loss.sum() / (final_mask.sum() + 1e-8)
        loss.backward()
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        self.optimizer.step()
        self.step += 1
        self.running_loss += loss.item()
        outputs_bin = (outputs > 0.5).float()
        correct = (outputs_bin == targets).float() * final_mask
        accuracy = correct.sum() / (final_mask.sum() + 1e-8)
        print_step = 200
        if self.step % print_step == 0:
            # print(f"Step {self.step}, Loss: {self.running_loss / print_step:.4f}")
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

    def train_batch_loop(self, batch_data, chunk_size=200):
        """
        DataLoaderから来た1バッチ分のデータを処理する関数
        batch_data: (inputs, targets, mask) from collate_fn
        inputs shape: (Batch, Max_Len, Input_Dim)
        """

        inputs_full, targets_full, mask_full = batch_data
        inputs_full = inputs_full.to("cuda", non_blocking=True)
        targets_full = targets_full.to("cuda", non_blocking=True)
        mask_full = mask_full.to("cuda", non_blocking=True)

        batch_size, max_len, _ = inputs_full.shape
        hidden_states = None
        
        total_loss = 0.0
        total_acc = 0.0
        count = 0

        # 2. 時間軸（Max_Len）に沿って chunk_size ずつ進むループ
        for t in range(0, max_len, chunk_size):
            # スライス範囲を計算
            end_t = min(t + chunk_size, max_len)
            
            # チャンクの切り出し
            input_chunk = inputs_full[:, t:end_t, :]
            target_chunk = targets_full[:, t:end_t, :]
            mask_chunk = mask_full[:, t:end_t, :]
            
            # このチャンクのmaskが全て0なら抜ける
            if mask_chunk.sum() == 0:
                break

            # train_step を実行
            loss, acc, hidden_states = self.train_step(
                input_chunk, 
                target_chunk, 
                mask_chunk, 
                hidden_states
            )
            
            total_loss += loss
            total_acc += acc
            count += 1
            
        return total_loss / max(count, 1), total_acc / max(count, 1)


    def save_model(self, filename):
        import os
        path = os.path.join(self.save_path, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    from joint_model import JointGRUNet
    import argparse

    parser = argparse.ArgumentParser(description="Train a joint network model")
    parser.add_argument("--epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint to load")
    args = parser.parse_args()

    sequence_length = setting.SEQUENCE_LENGTH
    input_size = setting.OBS_DIMENSION    # 観測は88次元
    hidden_size = setting.HIDDEN_SIZE
    output_size = setting.WHOLE_JOINT_NUM   # 19個の関節それぞれに故障があるかどうかを判断
    num_layers = setting.NUM_LAYERS      # GRUの層数
    chunk_size = setting.CHUNK_SIZE     # 学習時のチャンクサイズ
    max_grad_norm = setting.MAX_GRAD_NORM # 勾配クリッピングの最大ノルム
    batch_size = setting.BATCH_SIZE


    datasets = data.JointDataset(
                data_dir="processed_data",
                sequence_length=sequence_length,
                cache_in_memory=True
                )
    dataloader = torch.utils.data.DataLoader(
                        datasets,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=data.collate_fn_pad_batch,
                        num_workers=6,
                        pin_memory=True  # CPU→GPU転送を高速化
                    )

    failure_joint_list = [0, 1, 3, 4, 7, 8, 11, 12]

    param = {
        "hidden_size" : hidden_size,
        "num_layers" : num_layers,
        "chunk_size" : chunk_size,
        "failure_joint_list" : failure_joint_list,
        "max_grad_norm": max_grad_norm,
        "epochs" : args.epoch,
        "batch_size": batch_size
    }

    if args.checkpoint != "":
        print(f"Loading model from {args.checkpoint}")
        model = JointGRUNet(input_size, hidden_size, output_size, num_layers=num_layers).to("cuda")
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model = JointGRUNet(input_size, hidden_size, output_size, num_layers=num_layers).to("cuda")
    current_datetime = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"learning_log/{current_datetime}"
    trainer = Trainer(model, active_joint_indices=failure_joint_list, max_grad_norm=max_grad_norm, logdir=log_dir)
    total_loss = 0.0
    yaml.safe_dump(param, open(f"{log_dir}/training_setting.yaml", 'w'))

    print("Starting training... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    total_batches = 0
    for epoch in range(args.epoch):  # 10 epochs for demonstration
        start = time.time()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        for batch in dataloader:
            num_batches += batch[0].shape[0]
            loss, accuracy = trainer.train_batch_loop(batch_data=batch, chunk_size=chunk_size)
            total_loss += loss
            total_accuracy += accuracy
            
            total_batches += 1
            num_batches += 1
            
            current_mean_loss = total_loss / num_batches
            current_mean_accuracy = total_accuracy / num_batches
            trainer.writer.add_scalar("info / loss", current_mean_loss, total_batches)
            trainer.writer.add_scalar("info / accuracy", current_mean_accuracy, total_batches)

        end = time.time()
        trainer.writer.add_scalar("epoch_time", end - start, epoch)
        print(f"Epoch {epoch + 1} completed in {end - start:.2f} seconds")
        epoch_loss = total_loss / num_batches
        epoch_accuracy = total_accuracy / num_batches
        print(f"--> Loss: {epoch_loss:.4f}\n",f"--> Accuracy: {epoch_accuracy:.4f}")
        trainer.save_model(f"joint_net_epoch_{epoch + 1}.pth")
