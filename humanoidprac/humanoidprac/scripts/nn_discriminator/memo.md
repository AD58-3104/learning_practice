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