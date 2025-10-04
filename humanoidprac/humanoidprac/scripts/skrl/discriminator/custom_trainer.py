from skrl.trainers.torch import SequentialTrainer
import torch
import tqdm
import sys



class CustomTrainer(SequentialTrainer):
    def __init__(self,learned_agents = [] , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learned_agents = learned_agents

    def multi_act(self, states, agent, scope, timestep: int, timesteps: int):
        main_action = agent.act(states[scope[0] : scope[1]], timestep=timestep, timesteps=timesteps)[0]
        return self.learned_agents[main_action[0]].act(states[scope[0] : scope[1]], timestep=timestep, timesteps=timesteps)[0]
    
    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # discriminatorでモデルを選択 (離散アクション)
                model_selection_actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # バッチ化されたlearned agent推論で高速化
                selected_indices = model_selection_actions.long().squeeze(-1)  # [8192, 1] -> [8192]

                # 各モデルが選択された環境のマスクを作成
                model_0_mask = (selected_indices == 0)
                model_1_mask = (selected_indices == 1)

                env_input_actions = torch.zeros((states.shape[0], 19), device=states.device, dtype=torch.float32)

                # モデル0が選択された環境をバッチで処理
                if model_0_mask.any():
                    model_0_states = states[model_0_mask]
                    model_0_actions = self.learned_agents[0].act(model_0_states, timestep=timestep, timesteps=self.timesteps)[0]
                    env_input_actions[model_0_mask] = model_0_actions

                # モデル1が選択された環境をバッチで処理
                if model_1_mask.any():
                    model_1_states = states[model_1_mask]
                    model_1_actions = self.learned_agents[1].act(model_1_states, timestep=timestep, timesteps=self.timesteps)[0]
                    env_input_actions[model_1_mask] = model_1_actions

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(env_input_actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions (discriminatorの選択アクションを記録)
                # PPOが期待する形状に合わせる
                actions_for_record = selected_indices.unsqueeze(-1)  # [8192] -> [8192, 1]
                self.agents.record_transition(
                    states=states,
                    actions=actions_for_record,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

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

        # 学習済みモデルのエージェントはevalモードにする
        for learned_agent in self.learned_agents:
            learned_agent.set_running_mode("eval")

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
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                # ここで選択のアクションを取り出す
                actions = torch.vstack(
                    [
                        agent.act(states[scope[0] : scope[1]], timestep=timestep, timesteps=self.timesteps)[0]
                        # self.multi_act(states, agent, scope, timestep, self.timesteps)
                        for agent, scope in zip(self.agents, self.agents_scope)
                    ]
                )
                env_input_actions = torch.vstack(
                    [
                        self.learned_agents[actions[i][0]].act(states[scope[0] : scope[1]], timestep=timestep, timesteps=self.timesteps)[0]
                        for i, (agent, scope) in enumerate(zip(self.agents, self.agents_scope))
                    ]
                )
                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(env_input_actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
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