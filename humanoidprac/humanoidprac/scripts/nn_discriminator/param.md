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
