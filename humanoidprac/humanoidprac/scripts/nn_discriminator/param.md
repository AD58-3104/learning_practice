# パラメータチューニング

- 一回目
  - 2層 
  - 有効ジョイント [0, 1, 3, 4, 7, 8, 11, 12]
  - 30エポック
  - play時の入力10シーケンス毎で、被らせない.
  - 結果
```
Joint 0 accuracy: 6.68% , 215316.0  / 285939.0 [Detected failures / Real failures]
Joint 1 accuracy: 0.00% , 0.0  / 190600.0 [Detected failures / Real failures]
Joint 2 accuracy: 0.00% , 23391.0  / 0.0 [Detected failures / Real failures]
Joint 3 accuracy: 0.00% , 0.0  / 325255.0 [Detected failures / Real failures]
Joint 4 accuracy: 0.00% , 0.0  / 189910.0 [Detected failures / Real failures]
Joint 5 accuracy: 0.00% , 3820438.0  / 0.0 [Detected failures / Real failures]
Joint 6 accuracy: 0.00% , 1324690.0  / 0.0 [Detected failures / Real failures]
Joint 7 accuracy: 0.00% , 0.0  / 338417.0 [Detected failures / Real failures]
Joint 8 accuracy: 0.00% , 0.0  / 187848.0 [Detected failures / Real failures]
Joint 9 accuracy: 0.00% , 3630285.0  / 0.0 [Detected failures / Real failures]
Joint 10 accuracy: 0.00% , 79905.0  / 0.0 [Detected failures / Real failures]
Joint 11 accuracy: 28.86% , 26615.0  / 87901.0 [Detected failures / Real failures]
Joint 12 accuracy: 0.00% , 0.0  / 145051.0 [Detected failures / Real failures]
Joint 13 accuracy: 0.00% , 456472.0  / 0.0 [Detected failures / Real failures]
Joint 14 accuracy: 0.00% , 3300300.0  / 0.0 [Detected failures / Real failures]
Joint 15 accuracy: 0.00% , 2155422.0  / 0.0 [Detected failures / Real failures]
Joint 16 accuracy: 0.00% , 1931603.0  / 0.0 [Detected failures / Real failures]
Joint 17 accuracy: 0.00% , 618736.0  / 0.0 [Detected failures / Real failures]
Joint 18 accuracy: 0.00% , 1654116.0  / 0.0 [Detected failures / Real failures]
```
- 2回目
  - 2層 
  - 有効ジョイント [0, 1, 3, 4, 7, 8, 11, 12]
  - 30エポック
  - play時の入力1シーケンス毎
  - 結果
    - 上より若干確率が上がっている？

- 3回目
  - batch_size: 1024
  - chunk_size: 200
  - epochs: 100
  - failure_joint_list:
  - 0,1,3,4,7,8,11,12
  - hidden_size: 128
  - max_grad_norm: 1.0
  - num_layers: 2
```
Joint 0 accuracy: 0.00% , 0.0  / 209747.0 [Detected failures / Real failures]
Joint 1 accuracy: 0.00% , 0.0  / 135952.0 [Detected failures / Real failures]
Joint 2 accuracy: 0.00% , 3175134.0  / 0.0 [Detected failures / Real failures]
Joint 3 accuracy: 0.07% , 344.0  / 237299.0 [Detected failures / Real failures]
Joint 4 accuracy: 0.00% , 0.0  / 135403.0 [Detected failures / Real failures]
Joint 5 accuracy: 0.00% , 3225319.0  / 0.0 [Detected failures / Real failures]
Joint 6 accuracy: 0.00% , 1861623.0  / 0.0 [Detected failures / Real failures]
Joint 7 accuracy: 41.74% , 112986.0  / 237894.0 [Detected failures / Real failures]
Joint 8 accuracy: 44.43% , 73273.0  / 137338.0 [Detected failures / Real failures]
Joint 9 accuracy: 0.00% , 3157071.0  / 0.0 [Detected failures / Real failures]
Joint 10 accuracy: 0.00% , 3141009.0  / 0.0 [Detected failures / Real failures]
Joint 11 accuracy: 76.85% , 13096.0  / 17034.0 [Detected failures / Real failures]
Joint 12 accuracy: 81.17% , 73026.0  / 84393.0 [Detected failures / Real failures]
Joint 13 accuracy: 0.00% , 3249528.0  / 0.0 [Detected failures / Real failures]
Joint 14 accuracy: 0.00% , 122967.0  / 0.0 [Detected failures / Real failures]
Joint 15 accuracy: 0.00% , 2102714.0  / 0.0 [Detected failures / Real failures]
Joint 16 accuracy: 0.00% , 3193727.0  / 0.0 [Detected failures / Real failures]
Joint 17 accuracy: 0.00% , 1426814.0  / 0.0 [Detected failures / Real failures]
Joint 18 accuracy: 0.00% , 1821356.0  / 0.0 [Detected failures / Real failures]
```


# play時のチャンクサイズの違いによる差

- trainと同じ
- 10ステップ
  - 不思議な事にtrainと同じチャンクサイズよりもこちらの方が検出率が良い
```
Joint 0 accuracy: 1.40% , 17143.0  / 285939.0 [Detected failures / Real failures]
Joint 1 accuracy: 0.85% , 9131.0  / 190600.0 [Detected failures / Real failures]
Joint 2 accuracy: 0.00% , 1724968.0  / 0.0 [Detected failures / Real failures]
Joint 3 accuracy: 6.20% , 105470.0  / 325255.0 [Detected failures / Real failures]
Joint 4 accuracy: 0.05% , 286.0  / 189910.0 [Detected failures / Real failures]
Joint 5 accuracy: 0.00% , 2085600.0  / 0.0 [Detected failures / Real failures]
Joint 6 accuracy: 0.00% , 1168388.0  / 0.0 [Detected failures / Real failures]
Joint 7 accuracy: 44.62% , 235905.0  / 338417.0 [Detected failures / Real failures]
Joint 8 accuracy: 40.70% , 124580.0  / 187848.0 [Detected failures / Real failures]
Joint 9 accuracy: 0.00% , 1868475.0  / 0.0 [Detected failures / Real failures]
Joint 10 accuracy: 0.00% , 711062.0  / 0.0 [Detected failures / Real failures]
Joint 11 accuracy: 84.04% , 73921.0  / 87901.0 [Detected failures / Real failures]
Joint 12 accuracy: 85.03% , 145324.0  / 145051.0 [Detected failures / Real failures]
Joint 13 accuracy: 0.00% , 574619.0  / 0.0 [Detected failures / Real failures]
Joint 14 accuracy: 0.00% , 3642389.0  / 0.0 [Detected failures / Real failures]
Joint 15 accuracy: 0.00% , 2112243.0  / 0.0 [Detected failures / Real failures]
Joint 16 accuracy: 0.00% , 1198751.0  / 0.0 [Detected failures / Real failures]
Joint 17 accuracy: 0.00% , 1841240.0  / 0.0 [Detected failures / Real failures]
Joint 18 accuracy: 0.00% , 2442693.0  / 0.0 [Detected failures / Real failures]
```
- 1ステップ