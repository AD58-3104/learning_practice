import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# 環境の作成
env = gym.make('CartPole-v1')

bins_global = (3, 3, 6, 8)  # 離散化のビン数

# 状態空間を離散化する関数
def discretize_state(state, bins=(8, 8, 10, 12)):
    # CartPole環境の状態の範囲を手動で設定
    cart_position_bounds = [-2.4, 2.4]
    cart_velocity_bounds = [-4, 4]
    pole_angle_bounds = [-0.2095, 0.2095]
    pole_velocity_bounds = [-4, 4]
    
    # 状態を正規化
    cart_position = np.linspace(cart_position_bounds[0], cart_position_bounds[1], bins[0] + 1)
    cart_velocity = np.linspace(cart_velocity_bounds[0], cart_velocity_bounds[1], bins[1] + 1)
    pole_angle = np.linspace(pole_angle_bounds[0], pole_angle_bounds[1], bins[2] + 1)
    pole_velocity = np.linspace(pole_velocity_bounds[0], pole_velocity_bounds[1], bins[3] + 1)
    
    # 状態を離散化
    discrete_state = [
        np.digitize(state[0], cart_position),
        np.digitize(state[1], cart_velocity),
        np.digitize(state[2], pole_angle),
        np.digitize(state[3], pole_velocity)
    ]
    for i in range(len(bins)):
        if discrete_state[i] < 0:
            discrete_state[i] = 0
        if discrete_state[i] >= bins[i]:
            discrete_state[i] = bins[i] - 1
    return tuple(discrete_state)

# Q学習アルゴリズムの実装
def q_learning(env, episodes=5000, gamma=0.95, alpha=0.5, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):

    global bins_global

    # Q-tableの初期化
    q_table = np.zeros(bins_global + (env.action_space.n,))
    
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
        
        while not done:
            test_env.render()  # 環境を描画
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            next_state = discretize_state(next_state,bins_global)
            episode_reward += reward
            state = next_state
        time.sleep(0.5)
        print(f"test episode {episode}: reward = {episode_reward}")

# テスト実行
test_policy(q_table,5,2000)
env.close()