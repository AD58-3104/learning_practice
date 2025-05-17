import random
from collections import namedtuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchviz import make_dot
import math


class BinsConfig:
    __cart_pos: float
    __cart_vel: float
    __pole_angle: float
    __pole_vel: float

    def __init__(self, cart_pos=16, cart_vel=16, pole_angle=32, pole_vel=32):
        self.__cart_pos = cart_pos
        self.__cart_vel = cart_vel
        self.__pole_angle = pole_angle
        self.__pole_vel = pole_vel

    @property
    def cart_pos(self):
        return self.__cart_pos

    @property
    def cart_vel(self):
        return self.__cart_vel

    @property
    def pole_angle(self):
        return self.__pole_angle

    @property
    def pole_vel(self):
        return self.__pole_vel

    def get_bins(self):
        return (self.__cart_pos, self.__cart_vel, self.__pole_angle, self.__pole_vel)

    def bins_size(self):
        return len(self.get_bins())


# 状態空間を離散化する関数
def discretize_state(state, bins) -> np.array:
    # CartPole環境の状態の範囲を手動で設定
    cart_position_bounds = [-2.4, 2.4]
    cart_velocity_bounds = [-24, 24]
    pole_angle_bounds = [-0.2095, 0.2095]
    pole_velocity_bounds = [-4, 4]

    # 状態を正規化
    cart_position = np.linspace(
        cart_position_bounds[0], cart_position_bounds[1], bins.cart_pos + 1
    )
    cart_velocity = np.linspace(
        cart_velocity_bounds[0], cart_velocity_bounds[1], bins.cart_vel + 1
    )
    pole_angle = np.linspace(
        pole_angle_bounds[0], pole_angle_bounds[1], bins.pole_angle + 1
    )
    pole_velocity = np.linspace(
        pole_velocity_bounds[0], pole_velocity_bounds[1], bins.pole_vel + 1
    )

    # 状態を離散化
    discrete_state = np.array(
        [
            np.digitize(state[0], cart_position),
            np.digitize(state[1], cart_velocity),
            np.digitize(state[2], pole_angle),
            np.digitize(state[3], pole_velocity),
        ]
    )
    bin_arr = bins.get_bins()
    for i in range(bins.bins_size()):
        if discrete_state[i] < 0:
            discrete_state[i] = 0
        if discrete_state[i] >= bin_arr[i]:
            discrete_state[i] = bin_arr[i] - 1
    return discrete_state


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity: int = capacity
        self.memory: list = []
        self.position: int = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#  可視化
def visualize_model(model):
    dot = make_dot(model, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render("model")
    dot.view()


def select_action(env, policy_model, state: np.array):
    step_count = 0
    EPS_END = 0.05
    EPS_START = 0.9
    EPS_DECAY = 200

    while True:
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * step_count / EPS_DECAY) 
        # ε-greedy法で行動を選択
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = policy_model(torch.from_numpy(state)).argmax().item()
        step_count = step_count + 1
        yield action


if __name__ == "__main__":
    # カート位置、カート速度、ポール角度、ポール角速度のビン数 （ビン = 連続値を離散値に分割するときに利用する数の事）
    bins = BinsConfig()
    # 環境の作成
    env = gym.make("CartPole-v1")
    env.reset()

    device = torch.device("cpu")

    # モデル初期化

    policy_net = MLP(bins.bins_size(),env.action_space.n).to(device)
    target_net = MLP(bins.bins_size(),env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replaymemory = ReplayMemory(20000)
    batch_size = 128


    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)


    def optimize_model():
        if len(replaymemory) < batch_size:
            return
        transitions = replaymemory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # 状態をテンソルに変換
        state_batch = torch.cat([torch.tensor(s).unsqueeze(0) for s in batch.state])
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        next_state_batch = torch.cat([torch.tensor(s).unsqueeze(0) for s in batch.next_state])

        # Q値の計算
        state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        # 次の状態の最大Q値を計算
        next_state_values = target_net(next_state_batch).max(1)[0].detach()

        # 目標Q値を計算
        expected_state_action_values = (next_state_values * 0.99) + reward_batch

        # 損失を計算
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # 勾配をリセットし、逆伝播と最適化を実行
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
