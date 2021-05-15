

import gym
import pandas as pd
import tensorflow as tf
from a2c import Actor,Critic
from env import ArmEnv


OUTPUT_GRAPH = False
MAX_EPISODE =  500                     # 最大训练次数500
DISPLAY_REWARD_THRESHOLD = 200         # 如果奖励值大于这个阈值，显示图形界面环境
MAX_EP_STEPS = 2000                    # 每次训练最大步数
RENDER = False

LR_A = 0.001                           # actor 的学习率
LR_C = 0.01                            # critic 的学习率


# env = gym.make('CartPole-v0')        # gym环境
# env.seed(1)                          
# env = env.unwrapped

# N_F = env.observation_space.shape[0] # gym的state是四维
# N_A = env.action_space.n             # gym的action两个，向左、向右移动小车


env = ArmEnv()                         # Arm环境

N_F = env.state_dim                    # Arm的state
N_A = env.action_dim                   # Arm的动作维度


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)

sess.run(tf.global_variables_initializer())

res = []

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        # s_, r, done, info = env.step(a)
        s_, r, done = env.step(a)

        # if done: r = -20
        if done: 
            r = 0

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            # if running_reward > DISPLAY_REWARD_THRESHOLD: 
                # RENDER = False
            # print("episode:", i_episode, "  reward:", int(running_reward))
            
            print("episode:", i_episode, "  reward:", int(running_reward))
            res.append([i_episode, running_reward])
            break

pd.DataFrame(res,columns=['episode','a2c_reward']).to_csv('./a2c_reward.csv')