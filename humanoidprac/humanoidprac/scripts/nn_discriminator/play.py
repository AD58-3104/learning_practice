import torch
import nn_data
from nn_data import get_sequence_from_episode, get_episode_length
import time
import argparse
import tqdm
import setting
import torch.multiprocessing as mp
import csv
import os
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
    parser.add_argument("--finish_index", type=int, default=0, help="End index for evaluation data")
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

    # 混同行列用の変数
    true_positive = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")   # TP: 正しく故障と予測
    false_positive = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")  # FP: 誤って故障と予測
    true_negative = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")   # TN: 正しく正常と予測
    false_negative = torch.zeros(setting.WHOLE_JOINT_NUM).to("cuda")  # FN: 誤って正常と予測

    # 8x8混同行列用の変数（print_jointsで指定された8つの関節について）
    # confusion_8x8[i, j]: 関節print_joints[i]が実際に故障していて、関節print_joints[j]が故障と予測された回数
    print_joints = [1,4,8,12,0,3,7,11]
    confusion_8x8 = torch.zeros(len(print_joints), len(print_joints)).to("cuda")

    print("Starting evaluation... at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def print_result(target_joints = [i for i in range(setting.WHOLE_JOINT_NUM)]):
        for joint_id in target_joints:
            if total_real_failure[joint_id].item() != 0:
                joint_accuracy = total_correct[joint_id].item() / total_real_failure[joint_id].item()
            else:
                joint_accuracy = 0.0
            print(f"Joint {joint_id} accuracy: {joint_accuracy * 100:.2f}% , {total_detect[joint_id].item()}  / {total_real_failure[joint_id].item()} [Detected failures / Real failures]")

    def print_confusion_matrix(target_joints = [i for i in range(setting.WHOLE_JOINT_NUM)]):
        print("\n" + "="*80)
        print("Confusion Matrix Results")
        print("="*80)
        for joint_id in target_joints:
            tp = true_positive[joint_id].item()
            fp = false_positive[joint_id].item()
            tn = true_negative[joint_id].item()
            fn = false_negative[joint_id].item()

            print(f"\nJoint {joint_id}:")
            print(f"  Confusion Matrix:")
            print(f"                 Predicted Failure  Predicted Normal")
            print(f"  Actual Failure        {tp:6.0f}          {fn:6.0f}")
            print(f"  Actual Normal         {fp:6.0f}          {tn:6.0f}")

            # メトリクスの計算
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            print(f"  Accuracy:  {accuracy * 100:.2f}%")
            print(f"  Precision: {precision * 100:.2f}%")
            print(f"  Recall:    {recall * 100:.2f}%")
            print(f"  F1 Score:  {f1_score * 100:.2f}%")

    def save_confusion_matrix_to_csv(target_joints = [i for i in range(setting.WHOLE_JOINT_NUM)], output_file="confusion_matrix.csv"):
        """混同行列データをCSVファイルに保存"""
        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Joint_ID', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for joint_id in target_joints:
                tp = true_positive[joint_id].item()
                fp = false_positive[joint_id].item()
                tn = true_negative[joint_id].item()
                fn = false_negative[joint_id].item()

                # メトリクスの計算
                accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                writer.writerow({
                    'Joint_ID': joint_id,
                    'TP': int(tp),
                    'FP': int(fp),
                    'TN': int(tn),
                    'FN': int(fn),
                    'Accuracy': f"{accuracy:.6f}",
                    'Precision': f"{precision:.6f}",
                    'Recall': f"{recall:.6f}",
                    'F1_Score': f"{f1_score:.6f}"
                })

        print(f"\nConfusion matrix data saved to: {output_file}")

    with torch.no_grad():
        index = 0
        for episode in tqdm.tqdm(dataloader, desc="Evaluating"):
            index += 1
            if args.finish_index != 0 and index > args.finish_index:
                break
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

                # 混同行列の計算
                true_positive += (outputs & targets).long().sum(dim=(0, 1))      # TP: 両方が1
                false_positive += (outputs & ~targets).long().sum(dim=(0, 1))    # FP: 予測が1、実際は0
                true_negative += (~outputs & ~targets).long().sum(dim=(0, 1))    # TN: 両方が0
                false_negative += (~outputs & targets).long().sum(dim=(0, 1))    # FN: 予測が0、実際は1

                # 8x8混同行列の計算（ベクトル化版）
                # targets: (batch, seq_len, 19), outputs: (batch, seq_len, 19)
                # バッチとシーケンス次元を展開: (batch * seq_len, 19)
                targets_flat = targets.view(-1, setting.WHOLE_JOINT_NUM)
                outputs_flat = outputs.view(-1, setting.WHOLE_JOINT_NUM)

                # 8つの関節のみを抽出
                targets_8 = targets_flat[:, print_joints].float()  # (N, 8)
                outputs_8 = outputs_flat[:, print_joints].float()  # (N, 8)

                # 行列積で一気に計算: confusion_8x8[i, j] = sum_n(targets_8[n, i] * outputs_8[n, j])
                # targets_8.T @ outputs_8 -> (8, 8)
                # これで全ての(i, j)ペアについて、関節iが実際に故障していて関節jが故障と予測された回数を計算
                confusion_8x8 += targets_8.t() @ outputs_8

            batch_index += 1
            if batch_index % 300 == 0:
                print(f"Current accuracy after {batch_index} batches")
                print_result(print_joints)
                print_confusion_matrix(print_joints)
    print("Final evaluation results:")
    print_result(print_joints)
    print_confusion_matrix(print_joints)

    # 混同行列データをCSVに保存
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    csv_filename = f"confusion_matrix_{timestamp}.csv"
    save_confusion_matrix_to_csv(print_joints, csv_filename)

    # 8x8混同行列を表示
    print("\n" + "="*80)
    print("8x8 Confusion Matrix (Rows: Actual Failure, Columns: Predicted Failure)")
    print("="*80)
    print("Joint order:", print_joints)
    print("\nConfusion Matrix:")

    # ヘッダー行
    header = "        "
    for j_idx, j in enumerate(print_joints):
        header += f"  J{j:2d}  "
    print(header)

    # 各行を表示
    confusion_8x8_cpu = confusion_8x8.cpu()
    for i_idx, i in enumerate(print_joints):
        row_str = f"J{i:2d}  "
        for j_idx in range(len(print_joints)):
            row_str += f"{confusion_8x8_cpu[i_idx, j_idx].item():6.0f}  "
        print(row_str)

    # 8x8混同行列をCSVに保存
    csv_8x8_filename = f"confusion_matrix_8x8_{timestamp}.csv"
    with open(csv_8x8_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # ヘッダー行
        header_row = ['Actual\\Predicted'] + [f'Joint_{j}' for j in print_joints]
        writer.writerow(header_row)

        # データ行
        for i_idx, i in enumerate(print_joints):
            row = [f'Joint_{i}'] + [int(confusion_8x8_cpu[i_idx, j_idx].item()) for j_idx in range(len(print_joints))]
            writer.writerow(row)

    print(f"\n8x8 Confusion matrix saved to: {csv_8x8_filename}")