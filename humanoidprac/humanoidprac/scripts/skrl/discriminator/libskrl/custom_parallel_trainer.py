from typing import List, Optional, Union
import queue

import copy
import sys
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer

class StateHistoryQueue:
    def __init__(self, max_length: int, num_envs: int, device: str = "cuda"):
        self.max_length = max_length
        self.num_envs = num_envs
        self.device = device
        self.queue = queue.Queue(maxsize=max_length)
        # 10シーケンス以内にリセットされたかどうかを管理するテンソル
        self.reseted_queue = queue.Queue(maxsize=max_length)

    def append_state(self, state: torch.Tensor):
        if self.queue.full():
            self.queue.get()
        self.queue.put(state)

    def append_reseted(self, reseted: torch.Tensor):
        if self.reseted_queue.full():
            self.reseted_queue.get()
        self.reseted_queue.put(reseted)

    def get_state_sequence(self) -> torch.Tensor:
        states = list(self.queue.queue)
        return torch.stack(states, dim=0)  # shape: (sequence_length, num_envs, state_dim)

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
            return torch.ones((self.max_length,self.num_envs), dtype=torch.bool, device=self.device)
        # 各環境について、直近max_length個のリセット情報を取得し、リセットがあった場合Trueにする shape: (num_envs, 1)
        reseted_list = list(self.reseted_queue.queue)
        reseted_tensor = torch.stack(reseted_list, dim=0).any(dim=0)
        return reseted_tensor

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

        # 健康状態で動かすためのエージェント。これはactしか使わない
        self.health_agent = health_agent

        self.joint_num = 19  # ヒューマノイドの関節数

        # 状態履歴キューの初期化 これの長さはGRUのsequence_lengthに合わせる。なので10
        self.state_history_queue = StateHistoryQueue(max_length=10,num_envs=self.num_envs, device=self.device)
        
        # JointGRUNetモデルの読み込み
        # このモデルは出力を outputs > 0.6で二値化する必要があるので注意
        self.joint_gru_net = JointGRUNet(input_size=69, hidden_size=128, output_size=self.joint_num).to("cuda")
        self.joint_gru_net.load_state_dict(torch.load("model.pth"))
        self.joint_gru_net.eval()

        # これは後でいらなくなったら消してね
        if self.num_simultaneous_agents != 19:
            raise ValueError("len(Agents) must be same as joint number.")

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)
        self.health_agent.init(trainer_cfg=self.cfg)
        self.health_agent.set_running_mode("eval")  # 健康状態モデルはevalモードで固定

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
        states, infos = self.env.reset()

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
                    actions = self.health_agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                    next_states, _, terminated, truncated, _ = self.env.step(actions)
                    # 面倒なのでログは飛ばす
                    # 訓練も当然しない
                    # reset environments
                    if terminated.any() or truncated.any():
                        with torch.no_grad():
                            states, infos = self.env.reset()
                    else:
                        states = next_states
                    reseted_tensor = terminated | truncated
                    self.state_history_queue.append_reseted(reseted_tensor)
                continue

            with torch.no_grad():
                # shape (num_envs,)
                reseted_mask = self.state_history_queue.get_reseted_tensor()
                # shape (num_envs, 19)
                joint_failure = (self.joint_gru_net(self.state_history_queue.get_state_sequence()) > 0.6)
                # リセットされてすぐの環境は全ての関節が健康だとする
                joint_failure = joint_failure.masked_fill_(reseted_mask.unsqueeze(-1), False)
                
                # 各関節ごとに故障マスクテンソルを取得（id,tensor）
                joint_mask_tensors : list[tuple[int, torch.Tensor]] = []
                for joint_id in range(self.joint_num):
                    # 関節iが故障している環境についてはTrue、そうでなければFalseのマスクテンソルを作成
                    joint_mask = joint_failure[:, joint_id]  # shape: (num_envs,)
                    joint_mask_tensors.append((joint_id, joint_mask))

                # compute actions
                actions = torch.zeros((self.num_envs, self.env.action_space.shape[0]), device=self.device)
                for joint_id, joint_mask in joint_mask_tensors:
                    agent = self.agents[joint_id]
                    if joint_mask.any():
                        # 故障している環境については対応する関節に応じたエージェントで行動を計算
                        actions[joint_mask] = agent.act(
                            states[joint_mask], timestep=timestep, timesteps=self.timesteps
                        )[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # マスクを使って対応するエージェントに対して遷移を記録させる
                for joint_id, joint_mask in joint_mask_tensors:
                    self.agents[joint_id].record_transition(
                        states=states[joint_mask],
                        actions=actions[joint_mask],
                        rewards=rewards[joint_mask],
                        next_states=next_states[joint_mask],
                        terminated=terminated[joint_mask],
                        truncated=truncated[joint_mask],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states
            # リセット状態を状態履歴キューに追加
            reseted_tensor = terminated | truncated
            self.state_history_queue.append_reseted(reseted_tensor)

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

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # 状態履歴キューに現在の状態を追加
            self.state_history_queue.append_state(states)

            if not self.state_history_queue.ready():
                # ここに入るのは最初のsequense_length個のステップだけ
                with torch.no_grad():
                    # health_agentはinitの中でevalになっていて、その後trainには切り替えないのでこのままで良い
                    actions = self.health_agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                    next_states, _, terminated, truncated, _ = self.env.step(actions)
                    # 面倒なのでログは飛ばす
                    # 訓練も当然しない
                    # reset environments
                    if terminated.any() or truncated.any():
                        with torch.no_grad():
                            states, infos = self.env.reset()
                    else:
                        states = next_states
                    reseted_tensor = terminated | truncated
                    self.state_history_queue.append_reseted(reseted_tensor)
                continue

            with torch.no_grad():
                # shape (num_envs,)
                reseted_mask = self.state_history_queue.get_reseted_tensor()
                # shape (num_envs, 19)
                joint_failure = (self.joint_gru_net(self.state_history_queue.get_state_sequence()) > 0.6).to(torch.bool)
                # リセットされてすぐの環境は全ての関節が健康だとする
                joint_failure = joint_failure.masked_fill_(reseted_mask.unsqueeze(-1), False)
                
                # 各関節ごとに故障マスクテンソルを取得（id,tensor）
                joint_mask_tensors : list[tuple[int, torch.Tensor]] = []
                for joint_id in range(self.joint_num):
                    # 関節iが故障している環境についてはTrue、そうでなければFalseのマスクテンソルを作成
                    joint_mask = joint_failure[:, joint_id]  # shape: (num_envs,)
                    joint_mask_tensors.append((joint_id, joint_mask))

                # compute actions
                actions = torch.zeros((self.num_envs, self.env.action_space.shape[0]), device=self.device)
                for joint_id, joint_mask in joint_mask_tensors:
                    agent = self.agents[joint_id]
                    if joint_mask.any():
                        # 故障している環境については対応する関節に応じたエージェントで行動を計算
                        actions[joint_mask] = agent.act(
                            states[joint_mask], timestep=timestep, timesteps=self.timesteps
                        )[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states
            # リセット状態を状態履歴キューに追加
            reseted_tensor = terminated | truncated
            self.state_history_queue.append_reseted(reseted_tensor)
