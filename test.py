import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# 環境の作成
env = gym.make('CartPole-v1')

class BinsConfig:
    __cart_pos : float
    __cart_vel : float
    __pole_angle : float
    __pole_vel : float

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

# カート位置、カート速度、ポール角度、ポール角速度のビン数 （ビン = 連続値を離散値に分割するときに利用する数の事）
bins_global = BinsConfig()

# カートに与えられる入力は、左に移動、右に移動の2つだけ

# 状態空間を離散化する関数
def discretize_state(state, bins):
    # CartPole環境の状態の範囲を手動で設定
    cart_position_bounds = [-2.4, 2.4]
    cart_velocity_bounds = [-24, 24]
    pole_angle_bounds = [-0.2095, 0.2095]
    pole_velocity_bounds = [-4, 4]
    
    # 状態を正規化
    cart_position = np.linspace(cart_position_bounds[0], cart_position_bounds[1], bins.cart_pos + 1)
    cart_velocity = np.linspace(cart_velocity_bounds[0], cart_velocity_bounds[1], bins.cart_vel + 1)
    pole_angle = np.linspace(pole_angle_bounds[0], pole_angle_bounds[1], bins.pole_angle + 1)
    pole_velocity = np.linspace(pole_velocity_bounds[0], pole_velocity_bounds[1], bins.pole_vel + 1)
    
    # 状態を離散化
    discrete_state = [
        np.digitize(state[0], cart_position),
        np.digitize(state[1], cart_velocity),
        np.digitize(state[2], pole_angle),
        np.digitize(state[3], pole_velocity)
    ]
    bin_arr = bins.get_bins()
    for i in range(bins.bins_size()):
        if discrete_state[i] < 0:
            discrete_state[i] = 0
        if discrete_state[i] >= bin_arr[i]:
            discrete_state[i] = bin_arr[i] - 1
    return tuple(discrete_state)

# Q学習アルゴリズムの実装
def q_learning(env, episodes=2000, gamma=0.95, alpha=0.2, epsilon=0.5, epsilon_decay=0.99, min_epsilon=0.01):

    global bins_global

    # Q-tableの初期化
    q_table = np.zeros(bins_global.get_bins() + (env.action_space.n,))
    print("q_table shape -> ",q_table.shape)
    rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state,bins_global)
        done = False
        episode_reward = 0
        
        while not done:
            # ε-greedy方策による行動選択
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # 探索
            else:
                action = np.argmax(q_table[state])  # 活用
            
            # 行動を実行し、新しい状態と報酬を取得
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_state,bins_global)
            episode_reward += reward
            
            # Q値の更新
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * q_table[next_state][best_next_action])
            
            state = next_state
        
        # εの減衰
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(episode_reward)
        
        if episode % 100 == 0:
            print(f"episode {episode}: reward = {episode_reward}, ε = {epsilon:.2f}")
    
    return q_table, rewards

# 学習の実行
q_table, rewards = q_learning(env)

# 学習過程をプロット
plt.plot(rewards)
plt.xlabel('episode')
plt.ylabel('total reward')
plt.title('q learning learning curve')
plt.show()

# 学習したポリシーでテスト実行
def test_policy(q_table, episodes=10,max_steps=1000):
    global bins_global
    for episode in range(episodes):
        test_env = gym.make('CartPole-v1',render_mode='human', max_episode_steps=max_steps)
        state, _ = test_env.reset()
        state = discretize_state(state,bins_global)
        done = False
        episode_reward = 0
        stand_time = 0
        while not done:
            test_env.render()  # 環境を描画
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_state,bins_global)
            episode_reward += reward
            state = next_state
            stand_time += 1
        time.sleep(0.2)
        print(f"test episode {episode}: reward = {episode_reward} stand_time = {stand_time}")

# テスト実行
test_policy(q_table,5,2000)
env.close()