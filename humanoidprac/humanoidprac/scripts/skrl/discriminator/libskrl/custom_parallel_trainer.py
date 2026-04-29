from typing import List, Optional, Union
import queue
from collections import deque

import copy
import sys
import tqdm

import torch

from humanoidprac.tasks.manager_based.humanoidprac.mdp.events import EnvIdClassifier

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer

import os
nn_discriminator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../nn_discriminator"))
sys.path.insert(0, nn_discriminator_path)

from joint_model import JointGRUNet
import setting as nn_setting
import nn_data

sys.path.append("../../")
import logger


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


class JointAgentLoader:
    """関節名ごとに分かれた `agent_<joint_name>.pt` を 1 フォルダから読み込み、
    関節名キーの dict として保持するローダ。

    train_one.py + train_one_joint.sh で生成される 8 関節分のモデルを推論時に使う。
    """

    EXPECTED_JOINT_NAMES: List[str] = [
        "right_hip_yaw", "left_hip_yaw",
        "right_hip_roll", "left_hip_roll",
        "right_hip_pitch", "left_hip_pitch",
        "right_knee", "left_knee",
    ]

    def __init__(
        self,
        env,
        agent_cfg: dict,
        models_dir: str,
        file_prefix: str = "agent_",
        file_suffix: str = ".pt",
    ) -> None:
        self.env = env
        self.agent_cfg = agent_cfg
        self.models_dir = models_dir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.agents_by_joint_name: dict = {}
        self._load_all()

    def _scan_files(self) -> dict:
        import re
        pattern = re.compile(rf"^{re.escape(self.file_prefix)}(?P<name>.+){re.escape(self.file_suffix)}$")
        result: dict = {}
        for fname in os.listdir(self.models_dir):
            m = pattern.match(fname)
            if m is None:
                continue
            joint_name = m.group("name")
            result[joint_name] = os.path.join(self.models_dir, fname)
        return result

    def _load_all(self) -> None:
        found = self._scan_files()
        missing = [jn for jn in self.EXPECTED_JOINT_NAMES if jn not in found]
        if missing:
            raise FileNotFoundError(
                f"[JointAgentLoader] missing agents for joints: {missing} (in {self.models_dir})"
            )
        extra = [jn for jn in found.keys() if jn not in self.EXPECTED_JOINT_NAMES]
        if extra:
            print(f"[JointAgentLoader] WARN: ignoring unexpected files for: {extra}")

        for joint_name in self.EXPECTED_JOINT_NAMES:
            agent = self._build_ppo_agent(joint_name)
            agent.load(found[joint_name])
            agent.set_running_mode("eval")
            self.agents_by_joint_name[joint_name] = agent
        print(
            f"[JointAgentLoader] loaded {len(self.agents_by_joint_name)}/"
            f"{len(self.EXPECTED_JOINT_NAMES)} joint agents from {self.models_dir}"
        )

    def _build_ppo_agent(self, joint_name: str) -> Agent:
        """train_one.py:198-247 と同じ構造で PPO エージェントを構築する。"""
        from skrl.utils.model_instantiators.torch import shared_model
        from gymnasium.spaces import Box
        from skrl.agents.torch.ppo import PPO
        from skrl.resources.schedulers.torch import KLAdaptiveRL
        from skrl.memories.torch import RandomMemory
        from numpy import float32

        agent_cfg = copy.deepcopy(self.agent_cfg)
        agent_cfg["agent"]["experiment"]["experiment_name"] += f"_play_{joint_name}"
        agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveRL

        memory_size = agent_cfg["memory"]["memory_size"]
        if memory_size < 0:
            memory_size = agent_cfg["agent"]["rollouts"]
        memory = RandomMemory(
            memory_size=memory_size,
            num_envs=self.env.num_envs,
            device=self.env.device,
        )

        action_space = Box(
            low=-500.0, high=500.0,
            shape=self.env.action_space.shape, dtype=float32,
        )

        roles = ["policy", "value"]
        structure = [
            agent_cfg["models"]["policy"]["class"],
            agent_cfg["models"]["value"]["class"],
        ]
        parameters = [
            agent_cfg["models"]["policy"],
            agent_cfg["models"]["value"],
        ]
        instance_shared_models = shared_model(
            observation_space=self.env.observation_space,
            action_space=action_space,
            device=self.env.device,
            structure=structure,
            roles=roles,
            parameters=parameters,
        )
        models = {
            "policy": instance_shared_models,
            "value": instance_shared_models,
        }
        models["policy"].init_state_dict("policy")
        models["value"].init_state_dict("value")

        agent = PPO(
            models=models,
            memory=memory,
            cfg=agent_cfg["agent"],
            observation_space=self.env.observation_space,
            action_space=action_space,
            device=self.env.device,
        )
        return agent

    def get_agents_by_joint_name(self) -> dict:
        return self.agents_by_joint_name


class StateHistoryQueue:
    def __init__(self, max_length: int, num_envs: int, device: str = "cuda"):
        self.max_length = max_length
        self.num_envs = num_envs
        self.device = device
        self.queue = deque(maxlen=max_length)
        # 10シーケンス以内にリセットされたかどうかを管理するテンソル
        self.reseted_queue = deque(maxlen=max_length)

    def append_state(self, state: torch.Tensor):
        # state shape: (num_envs, state_dim)
        if self.ready():
            self.queue.popleft()
        self.queue.append(state.clone())

    def append_reseted(self, reseted: torch.Tensor):
        if self.ready():
            self.reseted_queue.popleft()
        self.reseted_queue.append(reseted.clone())

    def get_state_sequence(self) -> torch.Tensor:
        states = list(self.queue)
        ret = torch.stack(states, dim=0)   # shape (sequence_length, num_envs, state_dim)
        ret = ret.permute(1, 0, 2)  # shape (num_envs, sequence_length, state_dim)
        ret = ret.contiguous()
        return ret

    def ready(self) -> bool:
        """
        状態がGRUに入力可能かどうかを調べる。
        これがFalseになるのは最初のmax_length個のステップだけ。
        これがFalseの場合はGRUでの推論はスキップして、強制的に健康状態モデルで動かすことにする。
        """
        return len(self.queue) == self.max_length

    def get_reseted_tensor(self) -> torch.Tensor:
        """
        直近sequence_length分のリセット情報から書く環境についてリセットがあったかどうかを示すmask tensorを返す。
        マスクにはリセットがあった場合はTrueが入っている。
        shape(num_envs,)
        リセットがある場合は、その環境には十分なシーケンスが無いと判断するので、強制的に健康状態モデルを使うことになる。
        """
        if len(self.reseted_queue) < self.max_length:
            # 十分な情報がない場合はリセットされている = 健康状態モデルを強制的に使うとする
            return torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        # 各環境について、直近max_length個のリセット情報を取得し、リセットがあった場合Trueにする
        reseted_list = list(self.reseted_queue)  # (seq_len, num_envs)
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
        self.queue = deque(maxlen=max_length)

        # その環境で故障が発生しているかどうかを保持するテンソル。これはエピソードに渡って保持される
        self.failure_history_mask = torch.zeros((num_envs, self.joint_num), dtype=torch.bool, device=device)

    def full(self) -> bool:
        return len(self.queue) == self.max_length

    def append_failure(self, failure: torch.Tensor):
        # failure shape: (num_envs, joint_num)
        if self.full():
            self.queue.popleft()
        self.queue.append(failure.clone())

    def _get_current_failure_tensor(self) -> torch.Tensor:
        """
        直近sequence_length分の故障情報から、各環境の各関節が故障しているかどうかを示すtensorを返す
        これは、同じ関節が直近sequence_length回全て故障している場合にTrueとなる
        """
        # この処理は多分正しい
        failures = list(self.queue)
        ret = torch.stack(failures, dim=0)   # shape (sequence_length, num_envs, joint_num)
        ret = ret.permute(1, 0, 2)  # shape (num_envs, sequence_length, joint_num)
        return torch.all(ret, dim=1) # shape (num_envs, joint_num)

    def get_failure_joints(self) -> torch.Tensor:
        """
        今現在の故障状態マスクを返す。これは一度故障判定されたら、その環境がリセットされるまでTrueのまあ維持される
        shape(num_envs, joint_num)
        """
        failure_mask = self._get_current_failure_tensor()
        # ここでanyを使って、現在どの環境が関節故障していると判断されているのかを取得する
        # このマスクがTrueの環境は、そのエピソードで初めて故障されたと判断される環境となる。
        update_target_failure_mask = failure_mask.any(dim=1)  # shape(num_envs,)
        # 一度Trueになった環境はリセットされるまでTrueを維持する。ここでそのエピソード内で既にTrueになっている環境は更新対象から外す
        update_target_failure_mask[self.failure_history_mask.any(dim=1)] = False

        # 直近10回の履歴から壊れていると判断するマスクを用いて故障履歴マスクを更新する
        # これはまだTrueになっていない環境についてのみ更新する
        # shape(num_envs, joint_num) = shape(num_envs, joint_num)
        self.failure_history_mask[update_target_failure_mask] = failure_mask[update_target_failure_mask]
        return self.failure_history_mask

    def ready(self) -> bool:
        """
        故障履歴が全て埋まっているかどうかを調べる。
        これがFalseな場合は強制的に健康モデルで動かす
        """
        return self.full()
    
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
        single_policy_save_name: Optional[str] = None,
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

        self.data_preprocessor = nn_data.DataPreprocessor()
        self.data_preprocessor.load_scaler()

        # 健康状態で動かすためのエージェント。これはactしか使わない
        self.health_agent = health_agent

        self.joint_num = nn_setting.WHOLE_JOINT_NUM  # ヒューマノイドの関節数

        # 状態履歴キューの初期化 これの長さはGRUのsequence_lengthに合わせる。なので10
        self.state_history_queue = StateHistoryQueue(max_length=nn_setting.SEQUENCE_LENGTH,num_envs=self.num_envs, device=self.device)

        # 故障履歴キューの初期化
        self.failure_history_queue = FailureHistoryQueue(max_length=5,num_envs=self.num_envs, joint_num=self.joint_num, device=self.device)
        
        # JointGRUNetモデルの読み込み
        # このモデルは出力を outputs > 0.6で二値化する必要があるので注意
        # self.joint_gru_net = JointGRUNet(
        #                         input_size=nn_setting.OBS_DIMENSION, 
        #                         hidden_size=nn_setting.HIDDEN_SIZE, 
        #                         output_size=self.joint_num,
        #                         num_layers=nn_setting.NUM_LAYERS,
        #                         ).to("cuda")
        # self.joint_gru_net.load_state_dict(torch.load("joint_net_model.pth"))
        # self.joint_gru_net.eval()

        self.classifier = EnvIdClassifier(self.num_envs)

        # 単一エージェント (skrlにより agents=[single_agent] が単一インスタンスへアンラップされた場合) は
        # train_with_delayed_failure_info_single_policy 専用なのでクラス数との一致チェックをスキップする
        if isinstance(self.agents, list):
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
        # 単一ポリシー版の保存ファイル名 (例: target_joint の関節名を入れる)
        self.single_policy_save_name = single_policy_save_name

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

        detect_logger = logger.DiscriminatorTester(target_torque=50.0, num_envs=self.num_envs, joint_num=self.joint_num)

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
        states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする

        # 最初に故障履歴キューを全てFalseで埋めておく
        for _ in range(self.failure_history_queue.max_length):
            self.failure_history_queue.append_failure(torch.zeros((self.num_envs, self.joint_num), dtype=torch.bool, device=self.device))

        hidden_states = None # 隠れ層
        self.joint_gru_net.eval()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            # for agent in self.agents:
                # これは今の所全く何もしない
                # agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # 状態履歴キューに現在の状態を追加
            # self.state_history_queue.append_state(states)

            # if not self.state_history_queue.ready():
            #     # ここに入るのは最初のsequense_length個のステップだけ
            #     with torch.no_grad():
            #         # health_agentはinitの中でevalになっていて、その後trainには切り替えないのでこのままで良い
            #         actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
            #         # GRUに推論させて、隠れ層を更新する
            #         states_for_inf = self.data_preprocessor.process_tensor(states)
                    
            #         _, next_hidden_states = self.joint_gru_net(states_for_inf, hidden_states)
            #         hidden_states = next_hidden_states
            #         next_policy_states, _, terminated, truncated, _ = self.env.step(actions)
            #         next_states = self.env.obs_buf['state'] # 観測はobs_bufから'state'を取り出してそれを使うようにする
            #         # 面倒なのでログは飛ばす
            #         # 訓練も当然しない
            #         # reset environments
            #         if terminated.any() or truncated.any():
            #             with torch.no_grad():
            #                 policy_states, infos = self.env.reset()
            #                 states = self.env.obs_buf['state']  # 観測はobs_bufから'state'を取り出してそれを使うようにする
            #         else:
            #             states = next_states
            #             policy_states = next_policy_states
            #         reseted_tensor = terminated | truncated
            #         self.state_history_queue.append_reseted(reseted_tensor)
            #     continue

            with torch.no_grad():
                # 基本的に全てのテンソルは(num_envs, ...)の形状をしている事にする。
                # で、record_transitionの時だけ、classify_by_shapeで各クラス毎に分割して渡す。
                # shape (num_envs,)
                # reseted_mask = self.state_history_queue.get_reseted_tensor()
                # state_seq = self.state_history_queue.get_state_sequence()
                # shape (num_envs, 19)
                states_for_inf = self.data_preprocessor.process_tensor(states)
                joint_failure, hidden_states = (self.joint_gru_net(states_for_inf.unsqueeze(1), hidden_states))
                joint_failure = torch.sigmoid(joint_failure)
                joint_failure = (joint_failure > 0.5).long()  # 故障判定を二値化
                joint_mask = [2,5,6,9,10,13,14,15,16,17,18]
                joint_failure[:,:,joint_mask] = 0  # 関係ない関節は0にする

                # リセットされてすぐの環境は全ての関節が健康だとする
                # joint_failure = joint_failure.masked_fill_(reseted_mask.unsqueeze(-1), False)
                self.failure_history_queue.append_failure(joint_failure.squeeze(1))
                failure_joints_list = self.failure_history_queue.get_failure_joints()
                detect_logger.log(self.env, failure_joints_list)

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
                next_states = self.env.unwrapped.obs_buf['state'].clone() 

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
                detect_data  = detect_logger.get_data()
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    # 故障している環境のみの平均報酬を計算
                    if class_failure_mask.any():
                        agent_mean_reward = rewards[class_failure_mask].mean().item()
                        self.agents[class_id].track_data(f"Debug / mean_reward", agent_mean_reward)
                        # 故障している環境の数もログ
                        self.agents[class_id].track_data(f"Debug / num_failure_envs", class_failure_mask.sum().item())
                        for key, val in detect_data[class_id].items():
                            self.agents[class_id].track_data(f"Detect / {key}", val)


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
                    states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    reset_indices = torch.nonzero(reseted_tensor).squeeze(-1)
                    hidden_states[:, reset_indices, :] = 0.0
            else:
                states = next_states
                policy_states = next_policy_states
            # リセット状態を状態履歴キューに追加
            # self.state_history_queue.append_reseted(reseted_tensor)
            self.failure_history_queue.set_reseted(reseted_tensor)

            # モデルの定期保存
            if (timestep + 1) % self.model_save_interval == 0:
                # デフォルトの保存方法だと管理不能なのでこれで保存する
                self.save(self.model_save_dir)
        
        print("[INFO] Training completed.")
        self.save(self.model_save_dir)

    def train_with_delayed_failure_info(self) -> None:
        """Train the agents using the delayed fault notifier for model switching.

        GRU判別器の推論結果ではなく、change_random_joint_torque_with_delayed_notification
        が env._fault_notifier に登録した通知済み故障関節IDを直接用いてエージェントを切り替える。
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

        # change_random_joint_torque_with_delayed_notification が env に登録した notifier を取得
        fault_notifier = self.env.unwrapped._fault_notifier

        # 各クラスに属する joint_id の集合を事前計算 (num_envs,) 比較用
        class_joint_id_tensors = [
            torch.tensor(self.classifier.class_joint_id_list[cid], dtype=torch.long, device=self.device)
            for cid in range(self.classifier.num_of_classes)
        ]
        class_masks = [
            self.classifier.get_class_mask(cid) for cid in range(self.classifier.num_of_classes)
        ]

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            with torch.no_grad():
                # 遅延カウンタを進め、通知済み故障関節IDを取得する
                # shape (num_envs,), 未通知は-1
                fault_notifier.step()
                notified_faults = fault_notifier.get_notified_faults()

                # クラスごとに故障マスクテンソルを取得（id,tensor）
                # 通知済みの関節がそのクラスの担当関節に属し、かつ環境がそのクラスの範囲に属する場合にTrue
                class_failure_mask_tensors: list[tuple[int, torch.Tensor]] = []
                for class_id in range(self.classifier.num_of_classes):
                    joint_match = (notified_faults.unsqueeze(-1) == class_joint_id_tensors[class_id]).any(dim=-1)
                    class_failure_mask = class_masks[class_id] & joint_match
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
                        class_actions_list[class_id] = None

                # 最終的なアクションを決定（故障通知を受けた環境は対応クラスのエージェント、それ以外は健康状態）
                actions = health_actions.clone()
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    if class_failure_mask.any():
                        actions[class_failure_mask] = class_actions_list[class_id][class_failure_mask]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # 各エージェントで遷移を記録（全環境分だが、故障していない環境の報酬は0）
                agents_with_transitions = []
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    if class_failure_mask.any():
                        masked_rewards = torch.zeros_like(rewards)
                        masked_rewards[class_failure_mask] = rewards[class_failure_mask]

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
                    if class_failure_mask.any():
                        agent_mean_reward = rewards[class_failure_mask].mean().item()
                        self.agents[class_id].track_data("Debug / mean_reward", agent_mean_reward)
                        self.agents[class_id].track_data("Debug / num_failure_envs", class_failure_mask.sum().item())

                # 共通の環境情報もログ（オプション）
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction (record_transitionを呼び出したエージェントのみ)
            for class_id in agents_with_transitions:
                agent = self.agents[class_id]
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            # fault_notifier のリセットは sample_fault_notification_delay (mode="reset") が自動で行う
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
            else:
                policy_states = next_policy_states

            # モデルの定期保存
            if (timestep + 1) % self.model_save_interval == 0:
                self.save(self.model_save_dir)

        print("[INFO] Training completed.")
        self.save(self.model_save_dir)

    def train_with_delayed_failure_info_single_policy(self) -> None:
        """train_with_delayed_failure_info の単一ポリシー版

        故障クラスごとにエージェントを切り替えるのではなく、self.agents[0] のみを
        単一ポリシーとして全ての故障環境で利用する。
        通知済み故障関節IDが存在する環境ではこの単一ポリシーが動作し、
        それ以外の環境では健康状態エージェントが動作する。
        遷移の記録・post_interaction も self.agents[0] にのみ行う。
        """

        # skrl が agents=[single_agent] を単一インスタンスにアンラップした場合と、
        # 複数エージェントが渡されている場合の両方に対応
        single_agent = self.agents if not isinstance(self.agents, list) else self.agents[0]

        # set running mode
        single_agent.set_running_mode("train")

        # reset env
        policy_states, infos = self.env.reset()

        # change_random_joint_torque_with_delayed_notification が env に登録した notifier を取得
        fault_notifier = self.env.unwrapped._fault_notifier

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            with torch.no_grad():
                # 遅延カウンタを進め、通知済み故障関節IDを取得する
                # shape (num_envs,), 未通知は-1
                fault_notifier.step()
                notified_faults = fault_notifier.get_notified_faults()

                # 通知済み(故障判定済み)環境のマスク shape (num_envs,)
                failure_mask = notified_faults >= 0

                # まず健康状態エージェントで全環境のアクションを計算
                health_actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]

                # 故障環境がある場合は単一ポリシーのアクションも計算
                if failure_mask.any():
                    single_actions = single_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
                else:
                    single_actions = None

                # 最終的なアクションを決定（故障通知を受けた環境は単一ポリシー、それ以外は健康状態）
                actions = health_actions.clone()
                if failure_mask.any():
                    actions[failure_mask] = single_actions[failure_mask]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # 遷移を記録（故障環境のみ非ゼロ）
                recorded = False
                if failure_mask.any():
                    # 故障していない環境を0でマスクしたテンソルを作成
                    masked_states = torch.zeros_like(policy_states)
                    masked_states[failure_mask] = policy_states[failure_mask]
                    masked_actions = torch.zeros_like(single_actions)
                    masked_actions[failure_mask] = single_actions[failure_mask]
                    masked_rewards = torch.zeros_like(rewards)
                    masked_rewards[failure_mask] = rewards[failure_mask]
                    masked_next_states = torch.zeros_like(next_policy_states)
                    masked_next_states[failure_mask] = next_policy_states[failure_mask]
                    masked_terminated = torch.zeros_like(terminated)
                    masked_terminated[failure_mask] = terminated[failure_mask]
                    masked_truncated = torch.zeros_like(truncated)
                    masked_truncated[failure_mask] = truncated[failure_mask]

                    single_agent.record_transition(
                        states=masked_states,
                        actions=masked_actions,
                        rewards=masked_rewards,
                        next_states=masked_next_states,
                        terminated=masked_terminated,
                        truncated=masked_truncated,
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                    recorded = True

                    # ログ
                    agent_mean_reward = rewards[failure_mask].mean().item()
                    single_agent.track_data("Debug / mean_reward", agent_mean_reward)
                    single_agent.track_data("Debug / num_failure_envs", failure_mask.sum().item())

                # 共通の環境情報もログ（オプション）
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            single_agent.track_data(f"Info / {k}", v.item())

            # post-interaction (record_transitionを呼び出した場合のみ)
            if recorded:
                single_agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            # fault_notifier のリセットは sample_fault_notification_delay (mode="reset") が自動で行う
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
            else:
                policy_states = next_policy_states

            # モデルの定期保存
            if (timestep + 1) % self.model_save_interval == 0:
                self.save(self.model_save_dir)

        print("[INFO] Training completed.")
        self.save(self.model_save_dir)

    def eval_with_delayed_failure_info(self, expdata_logger: None, success_rate_logger: None, discriminator_tester: None) -> None:
        """Evaluate the agents using the delayed fault notifier for model switching.

        GRU判別器の推論結果ではなく、change_random_joint_torque_with_delayed_notification
        が env._fault_notifier に登録した通知済み故障関節IDを直接用いてエージェントを切り替える。
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

        # reset env
        policy_states, infos = self.env.reset()

        # change_random_joint_torque_with_delayed_notification が env に登録した notifier を取得
        fault_notifier = self.env.unwrapped._fault_notifier

        # 各クラスに属する joint_id の集合を事前計算 (num_envs,) 比較用
        class_joint_id_tensors = [
            torch.tensor(self.classifier.class_joint_id_list[cid], dtype=torch.long, device=self.device)
            for cid in range(self.classifier.num_of_classes)
        ]
        class_masks = [
            self.classifier.get_class_mask(cid) for cid in range(self.classifier.num_of_classes)
        ]

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            with torch.no_grad():
                # 遅延カウンタを進め、通知済み故障関節IDを取得する
                # shape (num_envs,), 未通知は-1
                fault_notifier.step()
                notified_faults = fault_notifier.get_notified_faults()

                # クラスごとに故障マスクテンソルを取得（id,tensor）
                # 通知済みの関節がそのクラスの担当関節に属し、かつ環境がそのクラスの範囲に属する場合にTrue
                class_failure_mask_tensors: list[tuple[int, torch.Tensor]] = []
                for class_id in range(self.classifier.num_of_classes):
                    joint_match = (notified_faults.unsqueeze(-1) == class_joint_id_tensors[class_id]).any(dim=-1)
                    class_failure_mask = class_masks[class_id] & joint_match
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
                        class_actions_list[class_id] = None

                # 最終的なアクションを決定（故障通知を受けた環境は対応クラスのエージェント、それ以外は健康状態）
                actions = health_actions.clone()
                for class_id, class_failure_mask in class_failure_mask_tensors:
                    if class_failure_mask.any():
                        actions[class_failure_mask] = class_actions_list[class_id][class_failure_mask]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # 判別器相当の故障関節マスクを構築（通知済み関節のみTrue）
                # shape (num_envs, joint_num)
                failure_joints_list = torch.zeros(
                    (self.num_envs, self.joint_num), dtype=torch.bool, device=self.device
                )
                notified_mask = notified_faults >= 0
                if notified_mask.any():
                    env_idx = torch.nonzero(notified_mask).squeeze(-1)
                    failure_joints_list[env_idx, notified_faults[notified_mask]] = True

                # ロギング
                if success_rate_logger is not None:
                    success_rate_logger.log(terminated, truncated)
                if discriminator_tester is not None:
                    discriminator_tester.log(self.env, failure_joints_list)
                if expdata_logger is not None:
                    if expdata_logger.log(self.env):
                        # これがTrueを返したら終了
                        break

                # render scene
                if not self.headless:
                    self.env.render()

            # reset environments
            # fault_notifier のリセットは sample_fault_notification_delay (mode="reset") が自動で行う
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
            else:
                policy_states = next_policy_states

        # 記録
        if success_rate_logger is not None:
            success_rate_logger.write_result()
        if discriminator_tester is not None:
            discriminator_tester.write_result()
        print("[INFO] Evaluation completed.")

    def eval_with_delayed_failure_info_by_joint_name(
        self,
        agents_by_joint_name: dict,
        expdata_logger=None,
        joint_survival_logger=None,
    ) -> None:
        """関節名キーの dict として渡されたエージェントを、遅延通知された故障関節IDで切り替える eval。

        EnvIdClassifier を一切参照せず、`fault_notifier.get_notified_faults()` の値を
        joint_id とみなして対応エージェントの行動を発行する。未通知環境は health_agent。
        """

        # joint_name -> joint_id を解決
        joint_names_tuple = self.env.unwrapped.scene["robot"].data.joint_names

        agents_by_id: dict = {}
        for joint_name, agent in agents_by_joint_name.items():
            try:
                jid = list(joint_names_tuple).index(joint_name)
            except ValueError:
                print(f"[eval_by_joint_name] WARN: joint_name '{joint_name}' is not in robot joint list. skipped.")
                continue
            agents_by_id[jid] = agent

        # set running mode
        self.health_agent.set_running_mode("eval")
        for agent in agents_by_joint_name.values():
            agent.set_running_mode("eval")

        # reset env
        policy_states, infos = self.env.reset()

        # change_random_joint_torque_with_delayed_notification が env に登録した notifier を取得
        fault_notifier = self.env.unwrapped._fault_notifier

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            with torch.no_grad():
                # 通知済み故障関節IDを取得 shape (num_envs,), 未通知は-1
                fault_notifier.step()
                notified_faults = fault_notifier.get_notified_faults()

                # 健康状態エージェントで全環境のアクションを計算
                health_actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
                actions = health_actions.clone()

                # joint_id 単位でアクションを上書き
                for joint_id, agent in agents_by_id.items():
                    mask = (notified_faults == joint_id)
                    if mask.any():
                        joint_actions = agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]
                        actions[mask] = joint_actions[mask]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # ロギング
                if joint_survival_logger is not None:
                    joint_survival_logger.log(self.env, terminated, truncated)
                if expdata_logger is not None:
                    if expdata_logger.log(self.env):
                        # 終了条件
                        break

            # reset environments
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
            else:
                policy_states = next_policy_states

        # 結果書き出し
        if joint_survival_logger is not None:
            joint_survival_logger.write_result()
        print("[INFO] Evaluation completed.")

    def eval_single_policy_for_joint(
        self,
        agent,
        joint_name: str,
        expdata_logger=None,
        joint_survival_logger=None,
    ) -> None:
        """単一ポリシーを 1 つの故障関節に対してテストする eval メソッド。

        env_cfg.events.change_random_joint_torque_with_delayed_notification.params.target_joint_cfg.joint_names
        が事前に `[joint_name]` のみへ絞られている前提。これにより故障イベントは常に joint_name のみを発火する。
        故障通知された env では `agent` のアクションが、それ以外では health_agent のアクションが採用される。

        実装は eval_with_delayed_failure_info_by_joint_name の単一エントリ dict 版として薄くラップする。
        """
        return self.eval_with_delayed_failure_info_by_joint_name(
            agents_by_joint_name={joint_name: agent},
            expdata_logger=expdata_logger,
            joint_survival_logger=joint_survival_logger,
        )

    def eval(self, expdata_logger: None, success_rate_logger: None, discriminator_tester: None) -> None:
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

        # reset env
        policy_states, infos = self.env.reset()
        states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする

        # 最初に故障履歴キューを全てFalseで埋めておく
        for _ in range(self.failure_history_queue.max_length):
            self.failure_history_queue.append_failure(torch.zeros((self.num_envs, self.joint_num), dtype=torch.bool, device=self.device))

        hidden_states = None # 隠れ層
        self.joint_gru_net.eval()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            with torch.no_grad():
                # shape (num_envs, 19)
                states_for_inf = self.data_preprocessor.process_tensor(states)
                joint_failure, hidden_states = (self.joint_gru_net(states_for_inf.unsqueeze(1), hidden_states))
                joint_failure = torch.sigmoid(joint_failure)
                joint_failure = (joint_failure > 0.5).long()  # 故障判定を二値化
                joint_mask = [2,5,6,9,10,13,14,15,16,17,18]
                joint_failure[:,:,joint_mask] = 0  # 関係ない関節は0にする

                # リセットされてすぐの環境は全ての関節が健康だとする
                # joint_failure = joint_failure.masked_fill_(reseted_mask.unsqueeze(-1), False)
                self.failure_history_queue.append_failure(joint_failure.squeeze(1))
                failure_joints_list = self.failure_history_queue.get_failure_joints()

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
                next_states = self.env.unwrapped.obs_buf['state'].clone() 

                if success_rate_logger is not None:
                    success_rate_logger.log(terminated, truncated)
                if discriminator_tester is not None:
                    discriminator_tester.log(self.env,failure_joints_list)
                if expdata_logger is not None:
                    if expdata_logger.log(self.env):
                        # これがTrueを返したら終了
                        break

                # render scene
                if not self.headless:
                    self.env.render()

            # reset environments
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
                    states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    reset_indices = torch.nonzero(reseted_tensor).squeeze(-1)
                    hidden_states[:, reset_indices, :] = 0.0
            else:
                states = next_states
                policy_states = next_policy_states
            # リセット状態を状態履歴キューに追加
            self.failure_history_queue.set_reseted(reseted_tensor)
        # 記録
        if success_rate_logger is not None:
            success_rate_logger.write_result()
        if discriminator_tester is not None:
            discriminator_tester.write_result()
        print("[INFO] Evaluation completed.")

    def eval_joint_model(self, expdata_logger: None, success_rate_logger: None, discriminator_tester: None, obs_logger: None, joint_torque_logger: None) -> None:
        # set running mode
        if self.num_simultaneous_agents > 1:
            self.health_agent.set_running_mode("eval")
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        hidden_states = None  # 隠れ層
        # reset env
        policy_states, infos = self.env.reset()
        states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする
        self.joint_gru_net.eval()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # 状態履歴キューに現在の状態を追加
            # self.state_history_queue.append_state(states)
            # print(states.dtype) torch.float32

            with torch.no_grad():
                # shape (num_envs, 19)
                # print(f"states shape: {states.unsqueeze(1).shape}")
                # state_seq = self.state_history_queue.get_state_sequence()
                self.joint_gru_net.eval()
                states = self.data_preprocessor.process_tensor(states)
                joint_failure, hidden_states = (self.joint_gru_net(states.unsqueeze(1), hidden=hidden_states))
                joint_failure = torch.sigmoid(joint_failure)
                joint_failure = (joint_failure > 0.5).long()  # 故障判定を二値化

                # まず健康状態エージェントで全環境のアクションを計算
                health_actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(health_actions)
                next_states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする
                # ロギング
                if success_rate_logger is not None:
                    success_rate_logger.log(terminated, truncated)
                if discriminator_tester is not None:
                    discriminator_tester.log(self.env,joint_failure.squeeze(1).bool())
                if obs_logger is not None:
                    obs_logger.log(self.env.common_step_counter, states, terminated, truncated)
                if joint_torque_logger is not None:
                    joint_torque_logger.log(self.env, terminated, truncated)
                if expdata_logger is not None:
                    if expdata_logger.log(self.env):
                        if success_rate_logger is not None:
                            success_rate_logger.write_result()
                        if discriminator_tester is not None:
                            discriminator_tester.write_result()
                        if obs_logger is not None:
                            obs_logger.close()
                        if joint_torque_logger is not None:
                            joint_torque_logger.close()
                        print("[INFO] Evaluation completed.")
                        break

                # render scene
                if not self.headless:
                    self.env.render()

            # reset environments
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
                    states = self.env.unwrapped.obs_buf['state'].clone() # 観測はobs_bufから'state'を取り出してそれを使うようにする
                    # reset_indices: リセットが必要な環境IDのリスト
                    reset_indices = torch.nonzero(reseted_tensor).squeeze(-1)
                    
                    # hidden_statesの形状は通常 (num_layers, batch_size, hidden_size)
                    # 該当する環境の隠れ状態を 0.0 にリセットする
                    # ※ .clone()が必要な場合がありますが、通常は直接代入でOKです
                    hidden_states[:, reset_indices, :] = 0.0
            else:
                states = next_states
                policy_states = next_policy_states
            # self.state_history_queue.append_reseted(reseted_tensor)
            self.failure_history_queue.set_reseted(reseted_tensor)

    def correct_data(self, obs_logger, joint_torque_logger) -> None:
        # set running mode
        if self.num_simultaneous_agents > 1:
            self.health_agent.set_running_mode("eval")
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # reset env
        policy_states, infos = self.env.reset()
        states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            with torch.no_grad():

                # まず健康状態エージェントで全環境のアクションを計算
                health_actions = self.health_agent.act(policy_states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_policy_states, rewards, terminated, truncated, infos = self.env.step(health_actions)
                next_states = self.env.unwrapped.obs_buf['state'].clone()  # 観測はobs_bufから'state'を取り出してそれを使うようにする
                if obs_logger is not None:
                    obs_logger.log(self.env.common_step_counter, states, terminated, truncated)
                if joint_torque_logger is not None:
                    joint_torque_logger.log(self.env, terminated, truncated)
                
                # render scene
                if not self.headless:
                    self.env.render()

            # reset environments
            reseted_tensor = terminated | truncated
            if reseted_tensor.any():
                with torch.no_grad():
                    policy_states, infos = self.env.reset()
                    states = self.env.unwrapped.obs_buf['state'].clone() # 観測はobs_bufから'state'を取り出してそれを使うようにする

            else:
                states = next_states
                policy_states = next_policy_states
        print("[INFO] Data correction completed.")
        obs_logger.close()
        joint_torque_logger.close()

    def save(self, dir_path: str) -> None:
        """
        学習したエージェントのモデルを保存する

        :param dir_path: モデルを保存するディレクトリパス
        :type dir_path: str
        """
        import os
        os.makedirs(dir_path, exist_ok=True)
        if isinstance(self.agents, list):
            # 複数エージェント: 各クラスのモデルをそれぞれ保存
            for joint_id, agent in enumerate(self.agents):
                agent.save(os.path.join(dir_path, f"agent_joint_{joint_id}.pt"))
        else:
            # 単一エージェント (single policy版): self.single_policy_save_name があればそれを使う
            filename = self.single_policy_save_name or "agent_single.pt"
            if not filename.endswith(".pt"):
                filename += ".pt"
            self.agents.save(os.path.join(dir_path, filename))
