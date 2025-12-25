# やり方

# 諸々の構築について

- 健康モデル作り ➡ Humanoidprac-v0-train (H1FlatEnvCfg)
- 健康モデルを使ってのデータ収集 ➡ Humanoidprac-nn-disc-data-correction (H1FlatEnvCfgCorrectLearningData)
- 学習したNNモデルを使っての各故障モデルの訓練 ➡ Parallel-failure-train-v0 (H1FlatEnvCfgRandomJointDebuff)

- **前提として**、最初の健康モデルがちゃんと歩く必要がある。
- データ収集タスクと故障モデル訓練タスクの違い ➡イベントでロギングをするか否か

## 健康モデルの訓練
以下のが良いのでは？
```bash 
labpython train.py --task Humanoidprac-v0-train --num_envs 8192 --headless agent.agent.experiment.directory="h1_flat/joint_experiment_ver3" env.events.change_joint_torque=null agent.trainer.timesteps=64000
```


## データの収集

このデータ収集は、基本的に1環境で行う。複数環境での収集には対応していない。

- 判別機用の観測を集める
  - logger.pyに実装されているDiscriminatorObsDataLoggerで収集する。ファイルはdiscriminator_obs.datに保存される。
- 故障イベントの情報を集める
  - change_random_joint_torqueイベントのloggingをオンにして集める
- 実行するスクリプト
  - skrl直下にある、play_hydra.pyを実行して行う。関節制限の対象は左右の足首以外とする.
  - これは健康状態で学習したモデルを使って、ランダムで関節故障を発生させるイベントが発生するタスクを行う事によって実行できる
  - この時、command_builderは使えない。以下のように指定してコマンドを実行する
  - labpython play_hydra.py --num_envs 1 --headless --finish_step 10000 --task Humanoidprac-nn-disc-data-correction --checkpoint <モデル> 

これを実行すると、discriminator_obs.datとjoint_torque_event_log.datが作られる。これをnn_discriminatorに持って行って学習する。

これを実行するとdiscriminator_obs.datとjoint_torque_event_log.datが作られる


## 実験履歴
集めたデータはdata_historyディレクトリに入れておく
- 0回目
  - 詳細は忘れた
  - データの保存先 disc_data_0

- 一回目
  - labpython play_hydra.py --num_envs 1 --headless --finish_step 10000 --task Humanoidprac-nn-disc-data-correction --checkpoint logs/skrl/h1_flat/joint_experiment_ver3/2025-09-18_11-45-54_ppo_torch_normal/checkpoints/best_agent.pt 
  - データの保存先 disc_data_1
  - ちなみに実装を間違えて片方の脚につき1つの関節が必ず壊れる環境になってしまった。なので複数関節故障であり、今回の学習では利用できないデータ
- 二回目
  - labpython play_hydra.py --num_envs 1 --headless --finish_step 10000 --task Humanoidprac-nn-disc-data-correction --checkpoint logs/skrl/h1_flat/joint_experiment_ver3/2025-09-18_11-45-54_ppo_torch_normal/checkpoints/best_agent.pt
- 三回目
  - エピソードリセット時にちゃんとトルクもリセットするようにした
  - labpython play_hydra.py --num_envs 1 --headless --finish_step 1600000 --task Humanoidprac-nn-disc-data-correction --checkpoint logs/skrl/h1_flat/joint_experiment_ver3/2025-09-18_11-45-54_ppo_torch_normal/checkpoints/best_agent.pt
- 四回目
  - 観測にトルクを追加した。合計88になった
  - labpython play_hydra.py --num_envs 1 --headless --finish_step 100000 --task Humanoidprac-nn-disc-data-correction --checkpoint logs/skrl/h1_flat/joint_experiment_ver3/2025-12-10_20-56-59_ppo_torch/checkpoints/best_agent.pt
- 五回目
  - 健康モデルを一回新しく学習しようとしたが、そもそも全然歩かない事態が発生。これを治す必要あり
  - 観測に追加したトルクの情報を結局無くした。トルク情報はNNにだけ渡すようにする
- 6回目
  - 色々直した
  - rm -rf nn_data/* ; labpython play_hydra.py --num_envs 4096 --headless --finish_step 1000 --task Humanoidprac-nn-disc-data-correction --checkpoint logs/skrl/h1_flat/joint_experiment_ver3/2025-12-12_19-21-51_ppo_torch/checkpoints/best_agent.pt

## 実行方法
1. 訓練データをnn_dataに入れる。
2. labpython data.py データを処理して学習用のデータに変換する
3. テスト用データのディレクトリに移動して、labpython data.pyを実行してテストデータを変換する
4. labpython train.py で学習を開始する
5. labpython play.pyでテストを実行する


## ストリーミングでの推論について
- 10個入れてそのたびに隠れ層更新は多分違うっぽい？色々な実装を見てみると、シーケンス長分バッファしてからそれを入力するような実装にしているやつはほぼ無い。なので多分止めた方が良いんだろう。一応AIによるとリセットいつしたいか見たいな基準によってはそれでも良いらしいが。
- ひとまずはskrlのGRUの実装を真似した方が良さそう。
- 隠れ層のリセットはいつ行う？
  - 


# TODO
- なぜ対して歩けないモデルが生成されるのかを探る
  - 👆単に報酬が悪そう。
  - でも不思議だ、変えた事と言えば観測を増やしたくらいで、他は全く変えていない。それなのに動作が悪くなることとかあるんや。観測が増えた分探索空間が増えたからとか？
  - 観測は元に戻してみた。nnに渡す用の観測クラスを作った


# 修正
明らかにデータセットがおかしい事を確認。data_stats.pyにより確認すると以下のようになった。
```
urrent accuracy after {batch_index} batches
Joint 0 failures 0.0 samples
Joint 1 failures 0.0 samples
Joint 2 failures 0.0 samples
Joint 3 failures 0.0 samples
Joint 4 failures 0.0 samples
Joint 5 failures 0.0 samples
Joint 6 failures 0.0 samples
Joint 7 failures 0.0 samples
Joint 8 failures 0.0 samples
Joint 9 failures 0.0 samples
Joint 10 failures 0.0 samples
Joint 11 failures 0.0 samples
Joint 12 failures 2763852.0 samples
Joint 13 failures 0.0 samples
Joint 14 failures 0.0 samples
Joint 15 failures 0.0 samples
Joint 16 failures 0.0 samples
Joint 17 failures 0.0 samples
Joint 18 failures 0.0 samples
Total samples evaluated: 3072000
Evaluating:   8%|██████████▏      
```

# dropoutの評価

dropoutあり
```
Final evaluation results:
Joint 0 accuracy: 90.26% , Real failures: 48044.0 samples
Joint 1 accuracy: 91.71% , Real failures: 40877.0 samples
Joint 2 accuracy: 100.00% , Real failures: 0.0 samples
Joint 3 accuracy: 89.72% , Real failures: 50689.0 samples
Joint 4 accuracy: 88.65% , Real failures: 55986.0 samples
Joint 5 accuracy: 100.00% , Real failures: 0.0 samples
Joint 6 accuracy: 100.00% , Real failures: 0.0 samples
Joint 7 accuracy: 92.34% , Real failures: 37749.0 samples
Joint 8 accuracy: 89.71% , Real failures: 50723.0 samples
Joint 9 accuracy: 100.00% , Real failures: 0.0 samples
Joint 10 accuracy: 100.00% , Real failures: 0.0 samples
Joint 11 accuracy: 97.45% , Real failures: 12590.0 samples
Joint 12 accuracy: 92.58% , Real failures: 36595.0 samples
Joint 13 accuracy: 100.00% , Real failures: 0.0 samples
Joint 14 accuracy: 100.00% , Real failures: 0.0 samples
Joint 15 accuracy: 100.00% , Real failures: 0.0 samples
Joint 16 accuracy: 100.00% , Real failures: 0.0 samples
Joint 17 accuracy: 100.00% , Real failures: 0.0 samples
Joint 18 accuracy: 100.00% , Real failures: 0.0 samples
```

# epoch数の評価
- 5エポック
```
Final evaluation results:
Joint 0 accuracy: 90.26% , Real failures: 48044.0 samples
Joint 1 accuracy: 91.71% , Real failures: 40877.0 samples
Joint 2 accuracy: 100.00% , Real failures: 0.0 samples
Joint 3 accuracy: 89.72% , Real failures: 50689.0 samples
Joint 4 accuracy: 88.65% , Real failures: 55986.0 samples
Joint 5 accuracy: 100.00% , Real failures: 0.0 samples
Joint 6 accuracy: 100.00% , Real failures: 0.0 samples
Joint 7 accuracy: 92.34% , Real failures: 37749.0 samples
Joint 8 accuracy: 89.71% , Real failures: 50723.0 samples
Joint 9 accuracy: 100.00% , Real failures: 0.0 samples
Joint 10 accuracy: 100.00% , Real failures: 0.0 samples
Joint 11 accuracy: 97.45% , Real failures: 12590.0 samples
Joint 12 accuracy: 92.58% , Real failures: 36595.0 samples
Joint 13 accuracy: 100.00% , Real failures: 0.0 samples
Joint 14 accuracy: 100.00% , Real failures: 0.0 samples
Joint 15 accuracy: 100.00% , Real failures: 0.0 samples
Joint 16 accuracy: 100.00% , Real failures: 0.0 samples
Joint 17 accuracy: 100.00% , Real failures: 0.0 samples
Joint 18 accuracy: 100.00% , Real failures: 0.0 samples
```

- 10エポック
```
Joint 0 accuracy: 90.26% , Real failures: 48044.0 samples
Joint 1 accuracy: 91.71% , Real failures: 40877.0 samples
Joint 2 accuracy: 100.00% , Real failures: 0.0 samples
Joint 3 accuracy: 89.72% , Real failures: 50689.0 samples
Joint 4 accuracy: 88.65% , Real failures: 55986.0 samples
Joint 5 accuracy: 100.00% , Real failures: 0.0 samples
Joint 6 accuracy: 100.00% , Real failures: 0.0 samples
Joint 7 accuracy: 92.34% , Real failures: 37749.0 samples
Joint 8 accuracy: 89.71% , Real failures: 50723.0 samples
Joint 9 accuracy: 100.00% , Real failures: 0.0 samples
Joint 10 accuracy: 100.00% , Real failures: 0.0 samples
Joint 11 accuracy: 97.45% , Real failures: 12590.0 samples
Joint 12 accuracy: 92.58% , Real failures: 36595.0 samples
Joint 13 accuracy: 100.00% , Real failures: 0.0 samples
Joint 14 accuracy: 100.00% , Real failures: 0.0 samples
Joint 15 accuracy: 100.00% , Real failures: 0.0 samples
Joint 16 accuracy: 100.00% , Real failures: 0.0 samples
Joint 17 accuracy: 100.00% , Real failures: 0.0 samples
Joint 18 accuracy: 100.00% , Real failures: 0.0 samples
```

# デバッグ

- 学習データを収集したタスク
  - Humanoidprac-nn-disc-data-correction
    - コンフィグ　skrl_ppo_cfg.yaml
    - クラス　H1FlatEnvCfgCorrectLearningData
- 実行時のタスク
  - Humanoidprac-v0-train-random-joint-debuff-play
    - コンフィグ　learned_agent_cfg.yaml
    - クラス　H1FlatEnvCfgRandomJointDebuff_PLAY
- 上記二つのコンフィグの違いは、trainerのステップとログディレクトリ程度。もしかして、seed nullが原因？　⬅ でもこれだとnn_discriminatorの中のplayでもダメになるはずなので、ここでは無い。
- クラスの違い
  - 実行時タスクには以下が含まれる
    - ジョイントのトルクを変更する際に、normalが1つ入っている。
    - 以下の設定が入っている。これは実行時タスクがH1FlatEnvCfg_PLAYを継承している事による
      - self.observations.policy.enable_corruption = False　　⬅これが関係ある？
      - self.events.base_external_force_torque = None
      - self.events.push_robot = None　
  - H1FlatEnvCfg_PLAYではなくH1FlatEnvCfgを継承するようにしてみたが、結果は変わらず全て0%。そうなると環境よりも実装が悪い？

- 実行時のタスクをH1FlatEnvCfgCorrectLearningDataにしてみる
  - してみたが、全ての関節において0%。

👆ここまで来ると、やっぱり条件では無くて実装の方が悪い？

## evalの実装のデバッグ
やること
- [ ] 機械学習のdataloaderでのやり方と同じやり方をする
- [ ] 最小の実装から初めて増やしていく
- [ ] そもそもnnのplayの実装が故障が起きている時に対する検出率の割合になっているか調べる。例えば、起きていない時に0が出ているのを正解としていないかどうか。
- [ ] 学習データの観測と実行データの観測が本当に同じかを確かめる
  - 学習データ  env.obs_buf['state']をファイルに書き出し
  - 実行時 env.obs_buf['state']を渡している

- [x] playでsequence長を1にして推論してみる。1にしてみた結果は以下のようになっており、推論自体は出来ているようだ。 
  - ➡ **ストリーミング推論で1つずつ入れているのは問題ではない。** 
  - ➡ **こちらではhiddenのリセットもしていないので多分それも問題ではない**
  - 多分問題はデータの読み込み方では無い
```
Joint 0 accuracy: 90.37% , Real failures: 48044.0 samples
Joint 1 accuracy: 91.80% , Real failures: 40877.0 samples
Joint 2 accuracy: 100.00% , Real failures: 0.0 samples
Joint 3 accuracy: 89.84% , Real failures: 50689.0 samples
Joint 4 accuracy: 88.77% , Real failures: 55986.0 samples
Joint 5 accuracy: 100.00% , Real failures: 0.0 samples
Joint 6 accuracy: 100.00% , Real failures: 0.0 samples
Joint 7 accuracy: 92.43% , Real failures: 37749.0 samples
Joint 8 accuracy: 89.83% , Real failures: 50723.0 samples
Joint 9 accuracy: 100.00% , Real failures: 0.0 samples
Joint 10 accuracy: 100.00% , Real failures: 0.0 samples
Joint 11 accuracy: 97.47% , Real failures: 12590.0 samples
Joint 12 accuracy: 92.66% , Real failures: 36595.0 samples
Joint 13 accuracy: 100.00% , Real failures: 0.0 samples
Joint 14 accuracy: 100.00% , Real failures: 0.0 samples
Joint 15 accuracy: 100.00% , Real failures: 0.0 samples
Joint 16 accuracy: 100.00% , Real failures: 0.0 samples
Joint 17 accuracy: 100.00% , Real failures: 0.0 samples
Joint 18 accuracy: 100.00% , Real failures: 0.0 samples
```

- [] 入力するシーケンスの順番が逆である説
reverseしてみた結果
```
[Class Success Rate Logger] Result written to class_success_rate.log
[Discriminator Tester] Joint 1: Success Rate 0.00% (0/414030)
[Discriminator Tester] Joint 4: Success Rate 0.00% (0/435173)
[Discriminator Tester] Joint 8: Success Rate 3.35% (13318/397303)
[Discriminator Tester] Joint 12: Success Rate 21.93% (72547/330815)
[Discriminator Tester] Joint 0: Success Rate 0.00% (0/414870)
[Discriminator Tester] Joint 3: Success Rate 0.00% (0/469964)
[Discriminator Tester] Joint 7: Success Rate 7.12% (32900/462193)
[Discriminator Tester] Joint 11: Success Rate 78.83% (85234/108129)
[Discriminator Tester] Joint 0: Total detected 9
[Discriminator Tester] Joint 1: Total detected 0
[Discriminator Tester] Joint 2: Total detected 0
[Discriminator Tester] Joint 3: Total detected 7
[Discriminator Tester] Joint 4: Total detected 0
[Discriminator Tester] Joint 5: Total detected 0
[Discriminator Tester] Joint 6: Total detected 0
[Discriminator Tester] Joint 7: Total detected 37368
[Discriminator Tester] Joint 8: Total detected 17703
[Discriminator Tester] Joint 9: Total detected 0
[Discriminator Tester] Joint 10: Total detected 0
[Discriminator Tester] Joint 11: Total detected 121029
[Discriminator Tester] Joint 12: Total detected 94665
[Discriminator Tester] Joint 13: Total detected 0
[Discriminator Tester] Joint 14: Total detected 0
[Discriminator Tester] Joint 15: Total detected 0
[Discriminator Tester] Joint 16: Total detected 0
[Discriminator Tester] Joint 17: Total detected 0
[Discriminator Tester] Joint 18: Total detected 0
[Discriminator Tester] Result written to discriminator_test_result.log
```
reverse無しの結果
```
[Class Success Rate Logger] Result written to class_success_rate.log
[Discriminator Tester] Joint 1: Success Rate 0.00% (0/413517)
[Discriminator Tester] Joint 4: Success Rate 0.00% (0/456174)
[Discriminator Tester] Joint 8: Success Rate 5.84% (24648/422249)
[Discriminator Tester] Joint 12: Success Rate 27.57% (95779/347351)
[Discriminator Tester] Joint 0: Success Rate 0.00% (0/466868)
[Discriminator Tester] Joint 3: Success Rate 0.00% (0/478289)
[Discriminator Tester] Joint 7: Success Rate 11.43% (50423/440974)
[Discriminator Tester] Joint 11: Success Rate 81.68% (93365/114312)
[Discriminator Tester] Joint 0: Total detected 1
[Discriminator Tester] Joint 1: Total detected 3
[Discriminator Tester] Joint 2: Total detected 0
[Discriminator Tester] Joint 3: Total detected 1
[Discriminator Tester] Joint 4: Total detected 3
[Discriminator Tester] Joint 5: Total detected 0
[Discriminator Tester] Joint 6: Total detected 0
[Discriminator Tester] Joint 7: Total detected 61022
[Discriminator Tester] Joint 8: Total detected 45383
[Discriminator Tester] Joint 9: Total detected 0
[Discriminator Tester] Joint 10: Total detected 0
[Discriminator Tester] Joint 11: Total detected 99849
[Discriminator Tester] Joint 12: Total detected 120761
[Discriminator Tester] Joint 13: Total detected 0
[Discriminator Tester] Joint 14: Total detected 0
[Discriminator Tester] Joint 15: Total detected 0
[Discriminator Tester] Joint 16: Total detected 0
[Discriminator Tester] Joint 17: Total detected 0
[Discriminator Tester] Joint 18: Total detected 0
```



## ラベルデバッグ
多分ラベルがおかしい。これのせいで検出ができていない。

実行時のエラーイベントの総数
```
[Discriminator Tester] Joint 1: Success Rate 0.00% (0/298355)
[Discriminator Tester] Joint 4: Success Rate 0.00% (0/331850)
[Discriminator Tester] Joint 8: Success Rate 6.03% (20063/332727)
[Discriminator Tester] Joint 12: Success Rate 30.19% (70706/234242)
[Discriminator Tester] Joint 0: Success Rate 0.00% (0/315106)
[Discriminator Tester] Joint 3: Success Rate 0.00% (0/331311)
[Discriminator Tester] Joint 7: Success Rate 12.55% (36004/286957)
[Discriminator Tester] Joint 11: Success Rate 82.09% (80285/97798)
[Discriminator Tester] Joint 0: Total detected 1
[Discriminator Tester] Joint 1: Total detected 5
[Discriminator Tester] Joint 2: Total detected 1
[Discriminator Tester] Joint 3: Total detected 1
[Discriminator Tester] Joint 4: Total detected 3
[Discriminator Tester] Joint 5: Total detected 1
[Discriminator Tester] Joint 6: Total detected 1
[Discriminator Tester] Joint 7: Total detected 44827
[Discriminator Tester] Joint 8: Total detected 34257
[Discriminator Tester] Joint 9: Total detected 1
[Discriminator Tester] Joint 10: Total detected 1
[Discriminator Tester] Joint 11: Total detected 85607
[Discriminator Tester] Joint 12: Total detected 96548
[Discriminator Tester] Joint 13: Total detected 1
[Discriminator Tester] Joint 14: Total detected 1
[Discriminator Tester] Joint 15: Total detected 1
[Discriminator Tester] Joint 16: Total detected 1
[Discriminator Tester] Joint 17: Total detected 1
[Discriminator Tester] Joint 18: Total detected 1
```


そもそもdata.effort_limitで大丈夫なのかは確認する必要がある。


```
Joint 0 accuracy: 90.17% , Real failures: 395240.0 samples
Joint 1 accuracy: 90.19% , Real failures: 394351.0 samples
Joint 2 accuracy: 100.00% , Real failures: 0.0 samples
Joint 3 accuracy: 89.91% , Real failures: 405547.0 samples
Joint 4 accuracy: 89.31% , Real failures: 429882.0 samples
Joint 5 accuracy: 100.00% , Real failures: 0.0 samples
Joint 6 accuracy: 100.00% , Real failures: 0.0 samples
Joint 7 accuracy: 91.87% , Real failures: 357001.0 samples
Joint 8 accuracy: 89.74% , Real failures: 422196.0 samples
Joint 9 accuracy: 100.00% , Real failures: 0.0 samples
Joint 10 accuracy: 100.00% , Real failures: 0.0 samples
Joint 11 accuracy: 99.56% , Real failures: 99372.0 samples
Joint 12 accuracy: 94.36% , Real failures: 278465.0 samples
```

多分故障のラベルが過剰になっている。消すべき所で消せていないかも。


# データセット設計
- ラベルデータ
  - 1ステップ毎に全ての環境分をそれぞれのファイルに保存する。ステップ数と終了も保存して、エピソードの切れ目が分かるようにしておく。その代わり、一回の試行で1つのファイルに保存する
- 観測データ
  - 1ステップ毎に全ての環境分の観測をそれぞれのファイルに保存する。それ以外に保存するのはステップ数だけ。1つのファイルで1回の試行
- データ形式
  - csvはデカくて効率が悪いので他のにした方が良さそう？
  - pandasは便利だが、pickleの方が良い？



[Discriminator Tester] Joint 1: Success Rate 0.00% (0/262559)
[Discriminator Tester] Joint 4: Success Rate 0.00% (0/298748)
[Discriminator Tester] Joint 8: Success Rate 0.00% (0/206834)
[Discriminator Tester] Joint 12: Success Rate 12.96% (25112/193762)
[Discriminator Tester] Joint 0: Success Rate 0.00% (0/255359)
[Discriminator Tester] Joint 3: Success Rate 0.00% (0/265909)
[Discriminator Tester] Joint 7: Success Rate 3.09% (6236/201541)
[Discriminator Tester] Joint 11: Success Rate 77.16% (53921/69886)