import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_confusion_matrix_from_csv(csv_file):
    """CSVファイルから混同行列データを読み込む"""
    joints_data = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            joint_data = {
                'joint_id': int(row['Joint_ID']),
                'tp': int(row['TP']),
                'fp': int(row['FP']),
                'tn': int(row['TN']),
                'fn': int(row['FN']),
                'accuracy': float(row['Accuracy']),
                'precision': float(row['Precision']),
                'recall': float(row['Recall']),
                'f1_score': float(row['F1_Score'])
            }
            joints_data.append(joint_data)

    return joints_data

def plot_confusion_matrices(joints_data, output_file='confusion_matrices.png'):
    """8つの関節の混同行列を8×8のサブプロットとして表示"""

    # 2x4のサブプロット配置（8つの関節用）
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Confusion Matrices for 8 Joints', fontsize=16, fontweight='bold')

    # サブプロットを1次元配列に変換
    axes = axes.flatten()

    for idx, joint_data in enumerate(joints_data):
        ax = axes[idx]
        joint_id = joint_data['joint_id']

        # 混同行列を作成 [[TP, FN], [FP, TN]]
        confusion_matrix = np.array([
            [joint_data['tp'], joint_data['fn']],
            [joint_data['fp'], joint_data['tn']]
        ])

        # ヒートマップを描画
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')

        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 軸ラベルとタイトルを設定
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted\nFailure', 'Predicted\nNormal'], fontsize=9)
        ax.set_yticklabels(['Actual\nFailure', 'Actual\nNormal'], fontsize=9)

        # タイトルにメトリクスを表示
        title = f'Joint {joint_id}\n'
        title += f'Acc: {joint_data["accuracy"]*100:.1f}% | '
        title += f'F1: {joint_data["f1_score"]*100:.1f}%'
        ax.set_title(title, fontsize=10, fontweight='bold')

        # セルに値を表示
        for i in range(2):
            for j in range(2):
                value = confusion_matrix[i, j]
                # 値が大きい場合は白文字、小さい場合は黒文字
                text_color = 'white' if value > confusion_matrix.max() / 2 else 'black'
                ax.text(j, i, f'{int(value)}',
                       ha='center', va='center',
                       color=text_color, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices plot saved to: {output_file}")
    plt.close()

def plot_single_confusion_matrix(joints_data, output_file='confusion_matrix_single.png'):
    """8つの関節のメトリクスを1つの大きなグラフにまとめて表示"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Performance Metrics Comparison for 8 Joints', fontsize=16, fontweight='bold')

    joint_ids = [jd['joint_id'] for jd in joints_data]
    accuracies = [jd['accuracy'] * 100 for jd in joints_data]
    precisions = [jd['precision'] * 100 for jd in joints_data]
    recalls = [jd['recall'] * 100 for jd in joints_data]
    f1_scores = [jd['f1_score'] * 100 for jd in joints_data]

    x_pos = np.arange(len(joint_ids))
    width = 0.6

    # Accuracy
    axes[0, 0].bar(x_pos, accuracies, width, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Joint ID', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(joint_ids)
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Precision
    axes[0, 1].bar(x_pos, precisions, width, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Joint ID', fontsize=12)
    axes[0, 1].set_ylabel('Precision (%)', fontsize=12)
    axes[0, 1].set_title('Precision', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(joint_ids)
    axes[0, 1].set_ylim([0, 100])
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Recall
    axes[1, 0].bar(x_pos, recalls, width, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Joint ID', fontsize=12)
    axes[1, 0].set_ylabel('Recall (%)', fontsize=12)
    axes[1, 0].set_title('Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(joint_ids)
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(axis='y', alpha=0.3)

    # F1 Score
    axes[1, 1].bar(x_pos, f1_scores, width, color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Joint ID', fontsize=12)
    axes[1, 1].set_ylabel('F1 Score (%)', fontsize=12)
    axes[1, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(joint_ids)
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison plot saved to: {output_file}")
    plt.close()

def load_8x8_confusion_matrix_from_csv(csv_file):
    """8x8混同行列CSVファイルを読み込む"""
    matrix = []
    joint_labels = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # ヘッダー行を読み込み
        joint_labels = [h.replace('Joint_', '') for h in headers[1:]]  # 'Joint_1' -> '1'

        for row in reader:
            # 最初の列（行ラベル）を除いて数値データを取得
            row_data = [int(val) for val in row[1:]]
            matrix.append(row_data)

    return np.array(matrix), joint_labels

def plot_8x8_confusion_matrix(matrix, joint_labels, output_file='confusion_matrix_8x8.png'):
    """8x8混同行列を1つの大きなヒートマップとして描画"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # ヒートマップを描画
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    # カラーバーを追加
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', rotation=270, labelpad=20, fontsize=12)

    # 軸ラベルを設定
    ax.set_xticks(np.arange(len(joint_labels)))
    ax.set_yticks(np.arange(len(joint_labels)))
    ax.set_xticklabels([f'J{label}' for label in joint_labels], fontsize=10)
    ax.set_yticklabels([f'J{label}' for label in joint_labels], fontsize=10)

    # 軸ラベルのタイトル
    ax.set_xlabel('Predicted Failure Joint', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Failure Joint', fontsize=14, fontweight='bold')
    ax.set_title('8x8 Confusion Matrix\n(Rows: Actual Failure, Columns: Predicted Failure)',
                 fontsize=16, fontweight='bold', pad=20)

    # セルに値を表示
    for i in range(len(joint_labels)):
        for j in range(len(joint_labels)):
            value = matrix[i, j]
            # 値が大きい場合は白文字、小さい場合は黒文字
            text_color = 'white' if value > matrix.max() / 2 else 'black'
            ax.text(j, i, f'{int(value)}',
                   ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')

    # グリッド線を追加
    ax.set_xticks(np.arange(len(joint_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(joint_labels)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"8x8 Confusion matrix plot saved to: {output_file}")
    plt.close()

def plot_8x8_normalized_confusion_matrix(matrix, joint_labels, output_file='confusion_matrix_8x8_normalized.png'):
    """8x8混同行列を正規化して描画（各行の合計を1に正規化）"""

    # 各行を正規化（実際に故障していた関節ごとに正規化）
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_sums,
                                  out=np.zeros_like(matrix, dtype=float),
                                  where=row_sums!=0)

    fig, ax = plt.subplots(figsize=(12, 10))

    # ヒートマップを描画
    im = ax.imshow(normalized_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # カラーバーを追加
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Ratio', rotation=270, labelpad=20, fontsize=12)

    # 軸ラベルを設定
    ax.set_xticks(np.arange(len(joint_labels)))
    ax.set_yticks(np.arange(len(joint_labels)))
    ax.set_xticklabels([f'J{label}' for label in joint_labels], fontsize=10)
    ax.set_yticklabels([f'J{label}' for label in joint_labels], fontsize=10)

    # 軸ラベルのタイトル
    ax.set_xlabel('Predicted Failure Joint', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Failure Joint', fontsize=14, fontweight='bold')
    ax.set_title('8x8 Confusion Matrix (Normalized by Row)\n(Rows: Actual Failure, Columns: Predicted Failure)',
                 fontsize=16, fontweight='bold', pad=20)

    # セルに値を表示（パーセンテージと実数値）
    for i in range(len(joint_labels)):
        for j in range(len(joint_labels)):
            value = normalized_matrix[i, j]
            count = matrix[i, j]
            # 値が大きい場合は白文字、小さい場合は黒文字
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value*100:.1f}%\n({int(count)})',
                   ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')

    # グリッド線を追加
    ax.set_xticks(np.arange(len(joint_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(joint_labels)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"8x8 Normalized confusion matrix plot saved to: {output_file}")
    plt.close()

def get_latest_csv(pattern="confusion_matrix_*.csv", exclude_pattern=None):
    """最新のCSVファイルを取得"""
    import glob
    csv_files = glob.glob(pattern)

    # 除外パターンがある場合は除外
    if exclude_pattern:
        exclude_files = set(glob.glob(exclude_pattern))
        csv_files = [f for f in csv_files if f not in exclude_files]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {pattern}")

    # 最新のファイルを取得
    latest_csv = max(csv_files, key=os.path.getmtime)
    return latest_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot confusion matrices from CSV")
    parser.add_argument("--csv", type=str, default=None, help="Path to the CSV file (default: latest confusion_matrix_*.csv)")
    parser.add_argument("--mode", type=str, default="individual", choices=["individual", "8x8"],
                        help="Plot mode: 'individual' for 8 separate 2x2 matrices, '8x8' for single 8x8 matrix")
    parser.add_argument("--output", type=str, default=None, help="Output file for confusion matrices")
    parser.add_argument("--metrics-output", type=str, default="metrics_comparison.png", help="Output file for metrics comparison (individual mode only)")
    parser.add_argument("--normalized", action="store_true", help="Also plot normalized 8x8 matrix (8x8 mode only)")
    args = parser.parse_args()

    if args.mode == "individual":
        # 個別の混同行列モード
        # CSVファイルを取得
        if args.csv is None:
            csv_file = get_latest_csv(pattern="confusion_matrix_*.csv", exclude_pattern="confusion_matrix_8x8_*.csv")
            print(f"Using latest CSV file: {csv_file}")
        else:
            csv_file = args.csv

        # 出力ファイル名のデフォルト
        output_file = args.output if args.output else "confusion_matrices.png"

        # CSVからデータを読み込み
        joints_data = load_confusion_matrix_from_csv(csv_file)

        # 混同行列をプロット
        plot_confusion_matrices(joints_data, output_file)

        # メトリクス比較をプロット
        plot_single_confusion_matrix(joints_data, args.metrics_output)

        print("\nPlotting completed successfully!")

    elif args.mode == "8x8":
        # 8x8混同行列モード
        # CSVファイルを取得
        if args.csv is None:
            csv_file = get_latest_csv(pattern="confusion_matrix_8x8_*.csv")
            print(f"Using latest 8x8 CSV file: {csv_file}")
        else:
            csv_file = args.csv

        # 出力ファイル名のデフォルト
        output_file = args.output if args.output else "confusion_matrix_8x8.png"

        # CSVから8x8混同行列を読み込み
        matrix, joint_labels = load_8x8_confusion_matrix_from_csv(csv_file)

        # 8x8混同行列をプロット
        plot_8x8_confusion_matrix(matrix, joint_labels, output_file)

        # 正規化版もプロット（オプション）
        if args.normalized:
            normalized_output = output_file.replace('.png', '_normalized.png')
            plot_8x8_normalized_confusion_matrix(matrix, joint_labels, normalized_output)

        print("\nPlotting completed successfully!")
