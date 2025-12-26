import torch
import numpy as np
import data # あなたのデータモジュール
import setting

def check_dataset_integrity():
    # Datasetをロード（学習時と同じ設定で）
    dataset = data.JointDataset(
        data_dir="processed_data",
        sequence_length=setting.SEQUENCE_LENGTH, # 例: 50
        cache_in_memory=True
    )
    
    print(f"Dataset Total Length: {len(dataset)}")
    if len(dataset) < 2:
        print("Data is too short to check overlap!")
        return

    for idx in range(400):
        # インデックス0と1（つまり時刻tとt+1）を取得
        sample0 = dataset[idx]['data'] # Shape: (Seq, 88)
        sample1 = dataset[idx+1]['data'] # Shape: (Seq, 88)


        # --- 検証1: 単純な値の確認 ---
        # sample0の [1行目] と sample1の [0行目] は、同じ時刻のデータのはずです
        # t=0,1,2... のとき、sample0は[0~49]、sample1は[1~50]
        
        # sample0の後ろ(Seq-1)個
        overlap_0 = sample0[1:] 
        # sample1の前(Seq-1)個
        overlap_1 = sample1[:-1]
        
        # 差分を計算
        diff = (overlap_0 - overlap_1).abs().max().item()
        
        if diff < 1e-5:
            continue
        else:
            print(f"--- Overlap Check (idx 0 vs idx 1) ---")
            print(f"Max Difference (Should be 0.0): {diff}")
            print("❌ Failure: Dataset is NOT sliding with stride=1.")
            print("   Possible reasons:")
            print("   1. Episode change: idx 0 and idx 1 might belong to different episodes.")
            print("   2. Data corruption: The raw .npz file format might be unexpected.")
            
            # エピソードの変わり目かチェック
            # Dataset内部の情報を覗き見る
            cumulative = dataset.sequence_cumulative_lengths
            print(f"Cumulative Lengths: {cumulative[:5]}")
            return 
            # もし [0, 1, 2...] のように増えているなら、各エピソードから1個しか取れていない（＝エピソードが短すぎる）
    print("✅ Success: Dataset is sliding correctly with stride=1.")
    print("   If DataLoader output looks wrong, check 'shuffle=True' or batch dimension handling.")

# check_dataloader_sliding.py
def check_sliding():
    datasets = data.JointDataset(
                        data_dir="test_data/processed_data",
                        sequence_length=10,
                        cache_in_memory=True
                        )
    dataloader = torch.utils.data.DataLoader(
                            datasets, 
                            batch_size=1, 
                            shuffle=False, 
                            collate_fn=data.collate_episodes,
                            num_workers=4
                        )

    prev_data = None
    for batch in dataloader:
        batch_data, batch_labels = batch  # batch_data shape: (batch_size, seq_len, feature_dim)
        if prev_data is None:
            prev_data = batch_data
            continue
        # バッチ内の index 0 (時刻 t) と index 1 (時刻 t+1) を比較
        # batch[0] の "1ステップ後" (1:50) と
        # batch[1] の "1ステップ前" (0:49) は完全に一致するはず
        print(batch_data.shape)
        seq_t0 = prev_data[:, 1:, 0]  # 0番目の関節
        seq_t1 = batch_data[:, :-1, 0]
        print(prev_data[:,:,0])
        print(batch_data[:,:,0])

        diff = (seq_t0 - seq_t1).abs().max().item()
        print(f"Sliding Diff (Batch[0] vs Batch[1]): {diff}")
        prev_data = batch_data

        if diff < 1e-5:
            print("✅ DataLoader is CORRECT. It is sliding perfectly.")
            print("   The 'wild fluctuation' you see is the ACTUAL signal content.")
        else:
            print("❌ DataLoader is broken.")

def check_collate():
    data_1 = torch.ones(5, 88)
    data_1[0,0] = 9.0
    label_1 = torch.ones(5,19)
    label_1[0,0] = 8.0

    data_2 = torch.ones(8, 88)
    label_2 = torch.ones(8,19)
    batch = [
        {'data': data_1, 'label': label_1},
        {'data': data_2, 'label': label_2}
    ]

    dt, label = data.collate_episodes(batch)
    print("Padded Data Shape:", dt.shape)  # Expect (2, 8, 88)
    print("Padded Label Shape:", label.shape)  # Expect (2, 8
    print(dt[0])
    print(label[0])
    print(dt[0].shape)
    print(label[0].shape)


if __name__ == "__main__":
    check_collate()