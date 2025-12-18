from typing import List, Optional, Union
import queue

import copy
import sys
import tqdm

import torch

from humanoidprac.tasks.manager_based.humanoidprac.mdp.events import EnvIdClassifier

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer

sys.path.append("../../../nn_discriminator")
from joint_model import JointGRUNet
import setting as nn_setting

class AgentModelLoader:
    def __init__(self, model_path: str, yaml_conf_path: str, env):
        """
        指定したパスにあるモデルを読み込んでエージェントにセットする
        """
        self.model_path = model_path
        self.agents = []
        import yaml
        self.yaml_conf = yaml.safe_load(open(yaml_conf_path))
        self.env = env
        self.load_models(model_path)


    def load_models(self, model_path: str) -> None:
        """
        指定したパスにあるモデルを全て読み込む。
        これは、trainerの方で利用するモデルの数と同じ数のモデルが保存されていることを前提とする。
        :param model_path: モデルが保存されているディレクトリパス
        """
        from skrl.utils.runner.torch import Runner
        rn = Runner(self.env, self.yaml_conf)
        print(f"[INFO] Loading agents from directory: {model_path}")
        model_pathes = self.get_model_pathes(model_path)
        for path in model_pathes:
            rn.agent.load(path)
            self.agents.append(rn.agent)
    
    @staticmethod
    def get_model_pathes(model_path: str) -> List[str]:
        import os
        import re
        files = os.listdir(model_path)
        sorted_filenames = sorted(files, key=lambda x: int(re.search(r'_(\d+)\.', x).group(1)))
        file_paths = []
        for filename in sorted_filenames:
            file_paths.append(os.path.join(model_path, filename))
        return file_paths


    def get_agents(self) -> List[Agent]:
        """
        Get the loaded agents.

        :return: The list of loaded agents.
        :rtype: List[skrl.agents.torch.Agent]
        """
        return self.agents

class StateHistoryQueue:
    def __init__(self, max_length: int, num_envs: int, device: str = "cuda"):
        self.max_length = max_length
        self.num_envs = num_envs
        self.device = device
        self.queue = queue.Queue(maxsize=max_length)
        # 10シーケンス以内にリセットされたかどうかを管理するテンソル
        self.reseted_queue = queue.Queue(maxsize=max_length)

    def append_state(self, state: torch.Tensor):
        # state shape: (num_envs, state_dim)
        if self.queue.full():
            self.queue.get()
        self.queue.put(state)

    def append_reseted(self, reseted: torch.Tensor):
        if self.reseted_queue.full():
            self.reseted_queue.get()
        self.reseted_queue.put(reseted)

    def get_state_sequence(self) -> torch.Tensor:
        states = list(self.queue.queue)
        ret = torch.stack(states, dim=0)   # shape (sequence_length, num_envs, state_dim)
        ret = ret.permute(1, 0, 2)  # shape (num_envs, sequence_length, state_dim)
        return ret

    def ready(self) -> bool:
        """
        状態がGRUに入力可能かどうかを調べる。
        これがFalseになるのは最初のmax_length個のステップだけ。
        これがFalseの場合はGRUでの推論はスキップして、強制的に健康状態モデルで動かすことにする。
        """
        return self.queue.full()

    def get_reseted_tensor(self) -> torch.Tensor:
        """
        直近sequence_length分のリセット情報から書く環境についてリセットがあったかどうかを示すmask tensorを返す。
        マスクにはリセットがあった場合はTrueが入っている。
        shape(num_envs,)
        リセットがある場合は、その環境には十分なシーケンスが無いと判断するので、強制的に健康状態モデルを使うことになる。
        """
        if self.reseted_queue.qsize() < self.max_length:
            # 十分な情報がない場合はリセットされている = 健康状態モデルを強制的に使うとする
            return torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        # 各環境について、直近max_length個のリセット情報を取得し、リセットがあった場合Trueにする
        reseted_list = list(self.reseted_queue.queue)  # (seq_len, num_envs)
        reseted_tensor = torch.stack(reseted_list, dim=0)
        # print(f"reseted_tensor shape: {reseted_tensor}")
        ret = reseted_tensor.any(dim=0).squeeze()
        # print(f"after any{ret}")
        # 形状を (num_envs,) に統一
        return ret

class FailureHistoryQueue:
    """
    関節故障履歴キュー
    これは関節の故障履歴を保持し、指定されたシーケンス長に渡って故障しているかどうかを判定するために利用される
    """
    def __init__(self, max_length: int, num_envs: int, joint_num: int, device: str = "cuda"):
        self.max_length = max_length
        self.num_envs = num_envs
        self.joint_num = joint_num
        self.device = device
        self.queue = queue.Queue(maxsize=max_length)

        # その環境で故障が発生しているかどうかを保持するテンソル。これはエピソードに渡って保持される
        self.failure_history_mask = torch.zeros((num_envs, self.joint_num), dtype=torch.bool, device=device)

    def append_failure(self, failure: torch.Tensor):
        # failure shape: (num_envs, joint_num)
        if self.queue.full():
            self.queue.get()
        self.queue.put(failure)

    def _get_current_failure_tensor(self) -> torch.Tensor:
        """
        直近sequence_length分の故障情報から、各環境の各関節が故障しているかどうかを示すtensorを返す
        """
        failures = list(self.queue.queue)
        ret = torch.stack(failures, dim=0)   # shape (sequence_length, num_envs, joint_num)
        ret = ret.permute(1, 0, 2)  # shape (num_envs, sequence_length, joint_num)
        return torch.all(ret, dim=1) # shape (num_envs, joint_num)

    def get_failure_joints(self) -> torch.Tensor:
        """
        今現在の故障状態マスクを返す。これは一度故障判定されたら、その環境がリセットされるまでTrueのまあ維持される
        shape(num_envs, joint_num)
        """
        failure_mask = self._get_current_failure_tensor()
        # 直近10回の履歴から壊れていると判断するマスク
        current_failure_mask = failure_mask.any(dim=1)  # shape(num_envs,)
        # 一度Trueになった環境はリセットされるまでTrueを維持する
        current_failure_mask[self.failure_history_mask.any(dim=1)] = False

        # 直近10回の履歴から壊れていると判断するマスクを用いて故障履歴マスクを更新する
        # これはまだTrueになっていない環境についてのみ更新する
        # shape(num_envs, joint_num) = shape(num_envs, joint_num)
        self.failure_history_mask[current_failure_mask] = failure_mask[current_failure_mask]
        return self.failure_history_mask

    def ready(self) -> bool:
        """
        故障履歴が全て埋まっているかどうかを調べる。
        これがFalseな場合は強制的に健康モデルで動かす
        """
        return self.queue.full()
    
    def set_reseted(self, reseted: torch.Tensor):
        """
        環境がリセットされた場合、その環境の故障状態をリセットする
        :param reseted: shape(num_envs,)のboolテンソル。リセットされた環境はFalseになる
        """
        self.failure_history_mask = self.failure_history_mask.masked_fill(reseted, False)



# fmt: off
# [start-config-dict-torch]
CUSTOM_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",       # key used to get and log environment info
    "stochastic_evaluation": False,      # whether to use actions rather than (deterministic) mean actions during evaluation
}
# [end-config-dict-torch]
# fmt: on


class CustomParallelAgentTrainer(Trainer):
    def __init__(
        self,
        env: Wrapper,
        health_agent: Agent,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Parallel agent trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See CUSTOM_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(CUSTOM_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # 便利なためコピーする
        self.num_envs = self.env.num_envs
        self.device = self.env.device

        # 健康状態で動かすためのエージェント。これはactしか使わない
        self.health_agent = health_agent

        self.joint_num = nn_setting.WHOLE_JOINT_NUM  # ヒューマノイドの関節数

        # 状態履歴キューの初期化 これの長さはGRUのsequence_lengthに合わせる。なので10
        self.state_history_queue = StateHistoryQueue(max_length=nn_setting.SEQUENCE_LENGTH,num_envs=self.num_envs, device=self.device)

        # 故障履歴キューの初期化 これの長さはGRUのsequence_lengthに合わせる。なので10
        self.failure_history_queue = FailureHistoryQueue(max_length=nn_setting.SEQUENCE_LENGTH,num_envs=self.num_envs, joint_num=self.joint_num, device=self.device)
        
        # JointGRUNetモデルの読み込み
        # このモデルは出力を outputs > 0.6で二値化する必要があるので注意
        self.joint_gru_net = JointGRUNet(input_size=nn_setting.OBS_DIMENSION, hidden_size=nn_setting.HIDDEN_SIZE, output_size=self.joint_num).to("cuda")
        self.joint_gru_net.load_state_dict(torch.load("joint_net_model.pth"))
        self.joint_gru_net.eval()

        self.classifier = EnvIdClassifier(self.num_envs)

        if len(self.agents) != self.classifier.num_of_classes:
            raise ValueError(f"Number of agents ({len(self.agents)}) does not match number of classes in EnvIdClassifier ({self.classifier.num_of_classes})")

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)
        self.health_agent.init(trainer_cfg=self.cfg)
        self.health_agent.set_running_mode("eval")  # 健康状態モデルはevalモードで固定
        import os
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_ppo")
        self.model_save_dir = os.path.join(self.cfg["trained_model_save_path"],current_time)
        self.model_save_interval = 3000 # timesteps毎に保存

    def train(self) -> None:
        """Train the agents sequentially

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_train()
            # multi-agent
            else:
                self.multi_agent_train()
            return

        # reset env
        policy_states, infos = self.env.reset()
        states = self.env.obs_buf['state']  # 観測はobs_bufから'state'を取り出してそれを使うようにする

        # 最初に故障履歴キューを全てFalseで埋めておく
        for _ in range(self.failure_history_queue.max_length):
            self.failure_history_queue.append_failure(torch.zeros((self.num_envs, self.joint_num), dtype=torch.bool, device=self.device))

        hidden_states = None # 隠れ層

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            # for agent in self.agents:
                # これは今の所全く何もしない
                # agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # 状態履歴キューに現在の状態を追加
            self.state_history_queue.append_state(states)

            if not self.state_history_queue.ready():
                # ここに入るのは最初のsequense_length個のステップだけ
                with torch.no_grad():
                    # health_agentはinitの中でevalになっていて、その後trainには切り替えないのでこのままで良い
                    actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
                    # GRUに推論させて、隠れ層を更新する
                    _, next_hidden_states = self.joint_gru_net(states, hidden_states)
                    hidden_states = next_hidden_states
                    next_policy_states, _, terminated, truncated, _ = self.env.step(actions)
                    next_states = self.env.obs_buf['state'] # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    # 面倒なのでログは飛ばす
                    # 訓練も当然しない
                    # reset environments
                    if terminated.any() or truncated.any():
                        with torch.no_grad():
                            policy_states, infos = self.env.reset()
                            states = self.env.obs_buf['state']  # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    else:
                        states = next_states
                        policy_states = next_policy_states
                    reseted_tensor = terminated | truncated
                    self.state_history_queue.append_reseted(reseted_tensor)
                continue

            with torch.no_grad():
                # 基本的に全てのテンソルは(num_envs, ...)の形状をしている事にする。
                # で、record_transitionの時だけ、classify_by_shapeで各クラス毎に分割して渡す。
                # shape (num_envs,)
                reseted_mask = self.state_history_queue.get_reseted_tensor()
                # state_seq = self.state_history_queue.get_state_sequence()
                # shape (num_envs, 19)
                joint_failure, hidden_states = (self.joint_gru_net(states, hidden_states))
                joint_failure = (joint_failure > 0.6).long()  # 故障判定を二値化


                # リセットされてすぐの環境は全ての関節が健康だとする
                joint_failure = joint_failure.masked_fill_(reseted_mask.unsqueeze(-1), False)
                self.failure_history_queue.append_failure(joint_failure)
                failure_joints_list = self.failure_history_queue.get_failure_joints()

                # if failure_joints_list.any():
                    # print(f"[DEBUG::train()] timestep {timestep} joint failure detected: {failure_joints_list}")
                
                # クラスごとに故障マスクテンソルを取得（id,tensor）
                class_failure_mask_tensors : list[tuple[int, torch.Tensor]] = []
                #  対応する関節のどれか1つが故障しているかを表すマスク(num_envs,)を得る。
                #  このマスクはそのクラスに属さない環境は全てFalseになっている。
                for class_id in range(self.classifier.num_of_classes):
                    # ジョイント故障テンソル(env, joint_num)を、そのクラスに対応するジョイントが故障しているかどうかのマスクに変換する
                    # 全環境サイズ(num_envs,)のマスクを作成
                    num_envs = failure_joints_list.shape[0]
                    class_failure_mask = torch.zeros(num_envs, dtype=torch.bool, device=failure_joints_list.device)

                    # そのクラスに属する環境のマスクを取得
                    class_mask = self.classifier.get_class_mask(class_id)

                    # そのクラスのジョイントが故障しているかをチェック
                    class_joint_failure = failure_joints_list[:, self.classifier.class_joint_id_list[class_id]].any(dim=1)

                    # クラスに属する環境のうち、故障している環境のみをTrueに
                    class_failure_mask[class_mask] = class_joint_failure[class_mask]

                    class_failure_mask_tensors.append((class_id, class_failure_mask))

                # まず健康状態エージェントで全環境のアクションを計算
                health_actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]

                # 各クラスエージェントのアクションを計算（全環境分）
                class_actions_list = [None for _ in range(self.classifier.num_of_classes)]
                for class_id in range(self.classifier.num_of_classes):
                    if class_failure_mask_tensors[class_id][1].any():
                        agent = self.agents[class_id]
                        class_actions_list[class_id] = agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
                    else:
                        # ない場合はとりあえずNoneにしとく
                        class_actions_list[class_id] = None

                # 最終的なアクションを決定（故障している関節のアクションを使用、それ以外は健康状態）
                actions = health_actions.clone()
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    if class_failure_mask.any():
                        actions[class_failure_mask] = class_actions_list[class_id][class_failure_mask]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.obs_buf['state']  # 観測はobs_bufから'state'を取り出してそれを使うようにする

                # render scene
                if not self.headless:
                    self.env.render()

                # 各エージェントで遷移を記録（全環境分だが、故障していない環境の報酬は0）
                # record_transitionを呼び出したエージェントのIDを記録
                agents_with_transitions = []
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    if class_failure_mask.any():
                        # 全環境分の報酬を作成し、故障している環境のみ報酬を付与
                        masked_rewards = torch.zeros_like(rewards)
                        masked_rewards[class_failure_mask] = rewards[class_failure_mask]

                        # 自クラス以外をマスクしたテンソルを取得（全環境分だが、他クラスは0）
                        classified_states = self.classifier.mask_other_classes(class_id, policy_states)
                        classified_actions = self.classifier.mask_other_classes(class_id, class_actions_list[class_id])
                        classified_rewards = self.classifier.mask_other_classes(class_id, masked_rewards)
                        classified_next_states = self.classifier.mask_other_classes(class_id, next_policy_states)
                        classified_terminated = self.classifier.mask_other_classes(class_id, terminated)
                        classified_truncated = self.classifier.mask_other_classes(class_id, truncated)

                        self.agents[class_id].record_transition(
                            states=classified_states,
                            actions=classified_actions,
                            rewards=classified_rewards,
                            next_states=classified_next_states,
                            terminated=classified_terminated,
                            truncated=classified_truncated,
                            infos=infos,
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )
                        agents_with_transitions.append(class_id)

                # log environment info (各エージェント固有の報酬をログ)
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    # 故障している環境のみの平均報酬を計算
                    if class_failure_mask.any():
                        agent_mean_reward = rewards[class_failure_mask].mean().item()
                        self.agents[class_id].track_data(f"Debug / mean_reward", agent_mean_reward)
                        # 故障している環境の数もログ
                        self.agents[class_id].track_data(f"Debug / num_failure_envs", class_failure_mask.sum().item())

                # 共通の環境情報もログ（オプション）
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            # 各エージェントに個別にログするのではなく、最初のエージェントだけにログ
                            # self.agents[0].track_data(f"Info / {k}", v.item())
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction (record_transitionを呼び出したエージェントのみ)
            for class_id in agents_with_transitions:
                agent = self.agents[class_id]
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
                    states = self.env.obs_buf['state'] # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    not_reseted = reseted_tensor.bool().logical_not().float()
                    mask = not_reseted.view(1, self.num_envs, 1)
                    hidden_states = hidden_states * mask
                    # print(f"[DEBUG] timestep {timestep} environments reset: {reseted_tensor}")
            else:
                states = next_states
                policy_states = next_policy_states
            # リセット状態を状態履歴キューに追加
            self.state_history_queue.append_reseted(reseted_tensor)
            self.failure_history_queue.set_reseted(reseted_tensor)

            # モデルの定期保存
            if (timestep + 1) % self.model_save_interval == 0:
                # デフォルトの保存方法だと管理不能なのでこれで保存する
                self.save(self.model_save_dir)
        
        print("[INFO] Training completed.")
        self.save(self.model_save_dir)

    def eval(self) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            self.health_agent.set_running_mode("eval")
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_eval()
            # multi-agent
            else:
                self.multi_agent_eval()
            return

        hidden_states = None  # 隠れ層
        # reset env
        policy_states, infos = self.env.reset()
        states = self.env.obs_buf['state']  # 観測はobs_bufから'state'を取り出してそれを使うようにする

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # 状態履歴キューに現在の状態を追加
            self.state_history_queue.append_state(states)

            if not self.state_history_queue.ready():
                # ここに入るのは最初のsequense_length個のステップだけ
                with torch.no_grad():
                    # health_agentはinitの中でevalになっていて、その後trainには切り替えないのでこのままで良い
                    actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
                    # GRUに推論させて、隠れ層を更新する
                    _, next_hidden_states = self.joint_gru_net(states, hidden_states)
                    hidden_states = next_hidden_states
                    next_policy_states, _, terminated, truncated, _ = self.env.step(actions)
                    next_states = self.env.obs_buf['state'] # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    # 面倒なのでログは飛ばす
                    # 訓練も当然しない
                    # reset environments
                    if terminated.any() or truncated.any():
                        with torch.no_grad():
                            policy_states, infos = self.env.reset()
                            states = self.env.obs_buf['state']  # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    else:
                        states = next_states
                        policy_states = next_policy_states
                    reseted_tensor = terminated | truncated
                    self.state_history_queue.append_reseted(reseted_tensor)
                continue

            with torch.no_grad():
                # 基本的に全てのテンソルは(num_envs, ...)の形状をしている事にする。
                # で、record_transitionの時だけ、classify_by_shapeで各クラス毎に分割して渡す。
                # shape (num_envs,)
                reseted_mask = self.state_history_queue.get_reseted_tensor()
                # state_seq = self.state_history_queue.get_state_sequence()
                # shape (num_envs, 19)
                joint_failure, hidden_states = (self.joint_gru_net(states, hidden_states))
                joint_failure = (joint_failure > 0.6).long()  # 故障判定を二値化


                # リセットされてすぐの環境は全ての関節が健康だとする
                joint_failure = joint_failure.masked_fill_(reseted_mask.unsqueeze(-1), False)
                self.failure_history_queue.append_failure(joint_failure)
                failure_joints_list = self.failure_history_queue.get_failure_joints()

                # if failure_joints_list.any():
                    # print(f"[DEBUG::train()] timestep {timestep} joint failure detected: {failure_joints_list}")
                
                # クラスごとに故障マスクテンソルを取得（id,tensor）
                class_failure_mask_tensors : list[tuple[int, torch.Tensor]] = []
                #  対応する関節のどれか1つが故障しているかを表すマスク(num_envs,)を得る。
                #  このマスクはそのクラスに属さない環境は全てFalseになっている。
                for class_id in range(self.classifier.num_of_classes):
                    # ジョイント故障テンソル(env, joint_num)を、そのクラスに対応するジョイントが故障しているかどうかのマスクに変換する
                    # 全環境サイズ(num_envs,)のマスクを作成
                    num_envs = failure_joints_list.shape[0]
                    class_failure_mask = torch.zeros(num_envs, dtype=torch.bool, device=failure_joints_list.device)

                    # そのクラスに属する環境のマスクを取得
                    class_mask = self.classifier.get_class_mask(class_id)

                    # そのクラスのジョイントが故障しているかをチェック
                    class_joint_failure = failure_joints_list[:, self.classifier.class_joint_id_list[class_id]].any(dim=1)

                    # クラスに属する環境のうち、故障している環境のみをTrueに
                    class_failure_mask[class_mask] = class_joint_failure[class_mask]

                    class_failure_mask_tensors.append((class_id, class_failure_mask))


                # まず健康状態エージェントで全環境のアクションを計算
                health_actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]

                # 各クラスエージェントのアクションを計算（全環境分）
                class_actions_list = [None for _ in range(self.classifier.num_of_classes)]
                for class_id in range(self.classifier.num_of_classes):
                    if class_failure_mask_tensors[class_id][1].any():
                        agent = self.agents[class_id]
                        class_actions_list[class_id] = agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
                    else:
                        # ない場合はとりあえずNoneにしとく
                        class_actions_list[class_id] = None

                # 最終的なアクションを決定（故障している関節のアクションを使用、それ以外は健康状態）
                actions = health_actions.clone()
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    if class_failure_mask.any():
                        actions[class_failure_mask] = class_actions_list[class_id][class_failure_mask]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(actions)
                next_states = self.env.obs_buf['state']  # 観測はobs_bufから'state'を取り出してそれを使うようにする

                # render scene
                if not self.headless:
                    self.env.render()

            # reset environments
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
                    states = self.env.obs_buf['state'] # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    not_reseted = reseted_tensor.bool().logical_not().float()
                    mask = not_reseted.view(1, self.num_envs, 1)
                    hidden_states = hidden_states * mask
                    # print(f"[DEBUG] timestep {timestep} environments reset: {reseted_tensor}")
            else:
                states = next_states
                policy_states = next_policy_states
            # リセット状態を状態履歴キューに追加
            self.state_history_queue.append_reseted(reseted_tensor)
            self.failure_history_queue.set_reseted(reseted_tensor)

    def save(self, dir_path: str) -> None:
        """
        学習したエージェントのモデルを保存する

        :param dir_path: モデルを保存するディレクトリパス
        :type dir_path: str
        """
        # 各エージェントのやつを全部保存する
        import os
        os.makedirs(dir_path, exist_ok=True)
        for joint_id, agent in enumerate(self.agents):
            agent.save(os.path.join(dir_path, f"agent_joint_{joint_id}.pt"))
