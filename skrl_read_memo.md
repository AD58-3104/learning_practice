# Agent
PPOなどを含む全てのエージェントはbase.pyにあるAgentクラスを継承している。
どうでも良いが、これxpu使えるpytorchのバージョンでskrl動かしたらxpuで動くんでは？
```python
        self.device = config.torch.parse_device(device)
```
## PPO
以下はPPOをagentを読んだメモ

### act()
これはmodelのactを呼び出しているだけ。アクションを実行している。確率論的なアクションを取り出す。

### record_transition()
環境の遷移を保存する。保存するのは以下の通り。まあSARS。
- state
- action
- reward
- next_state
- terminated
- truncated
- infos
- timestep
- timesteps

### post_interaction()
trainモードで内部で_updateを呼び出したのちにevalに戻す。
```python
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)
```

### _update()
これがactと並んで重要なやつだと思う。この中でgaeを計算する関数が定義されている。
処理の流れ
1. まずvalueのactを、__current_next_states(これは今の状態に対して取り出したアクションを適用して遷移した状態)に対して実行し、アクション適用後の価値を計算する。勿論この時はevalにして実行する。
2. 1.で計算した次状態の価値と、今の状態の価値と報酬を使ってgaeを計算する。これを計算すると、advantageとreturns(advantage + valueなので、行動価値？)が返ってくる
3. values,returns,advantagesをmemoryに記憶する。
4. memoryからバッチ分の記録(SARS等の情報)を取り出す
5. 取り出したバッチを使って学習を回す
   1. 取り出したactionとstateでpolicyにactさせる。それでlog_probを取り出す。他のactionとかは無視する。
   2. このlog_probとバッチから取り出したlog_probからkl divergenceを計算する。これがthreshouldより大きい場合はbreakする。一定以上の勾配は更新しないというアレの事だろう。
   3. entropy_lossも計算している。
   4. うーんsurrogateって出てきたからこれ多分PPOの代理関数の計算のやつだ。まだ理解してないやつだ...
```python
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
```
    5. value_lossの計算をする。なんかF.mse_lossというやつで計算しているが、これはPPOのなんかか...?
       1. 平均二乗誤差の事だった。別に普通だ。
    6. optimizerとscalarを使って最適化
    7. なんかdistributedの場合のよくわからん処理
    8. 勾配のクリップ？これPPOか？それとも普通に更新の最大値を制限するだけの単純な処理か？
    9.  lerning_rate_schedulerで学習率を上手くやる。まあこれはいいでしょう。

ちなみにこの中でサンプリングしてポリシーの更新を行うやつはオンポリシー学習であるにも関わらず、オフポリシー学習みたいな事をしている。これは重点サンプリングによって可能になっている。

### Policy
これはmodelsの中に入っているやつ。gausianとかその辺に分かれる


# TODO
- この中でのvalueが行動価値と状態価値のどちらを指すのか？まあ状態価値の可能性の方が多分高い
- actのとこにあるlog_probって何だ...これは...。これを理解しないと先に進めなさそうだ。
  - これはtorch.distributionsに含まれる分布モデルが持つやつ。
  - ランダムサンプリングは逆伝播出来ない。なので、逆伝播するための代理関数が必要で、それを作る典型的な方法は2つある。1つがREINFORCE/スコア関数推定器/尤度比推定器で、もう1つが経路微分推定器である。REINFORCEは強化学習における方策勾配法の基礎であり、後者は変分オートエンコーダの再パラメータ化トリックで一般的に用いられる。スコア関数はf(x)のサンプルのみを必要として、経路微分では導関数f'(x)を求める必要がある。
  - 確率密度関数がそのパラメータに対して微分可能であればsample()とlog_prob()を実装するだけで良い。
    - $\Delta{\theta}=\alpha r\frac{\delta logp(a|\pi^\theta(s))}{\delta\theta}$
    - ここで、$\theta$はパラメータで、$\alpha$は学習率、rは報酬で、$p(a|\pi^\theta(s))$はアクションaを状態sで与えられた方策$\pi^\theta$の元で実行する確率
    - 上記から、実践的にはネットワークの出力からアクションをサンプルして、それを環境に適用して同等の損失関数を構築すればよい。上記のルールは勾配上昇法を想定しているため、勾配降下法に対応させるために下の例ではマイナスを掛けている。

```python
probs = policy_network(state)
# Note that this is equivalent to what used to be called multinomial
m = Categorical(probs)
action = m.sample()
next_state, reward = env.step(action)
loss = -m.log_prob(action) * reward
loss.backward()
```

- あ、これそういう事か。まず、アクションをサンプリングする。これは、上の式のlogp()の中身のアクションがサンプリングされる確率の所を表している。で、この取り出したアクションで報酬を計算する、これは当然rを表す。で、多分log_probは$\delta logp()$になっている。それらを掛け合わせたのがlossになってるわけだ。
    - ちなみに確率密度関数を微分してないけどいいのかこれは。
    - 👆バカすぎた。これここで微分しなくても、loss.backward()をした時に自動微分されるからマジでこれで良いんだ。賢い実装だ。
- entropy_lossが何かを理解する
- PPOの代理関数とかその辺のやつをちゃんと理解する