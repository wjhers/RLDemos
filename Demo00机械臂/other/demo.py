'''
训练脚本

'''
# 多智能体环境
import pandas as pd
import tensorflow as tf
import numpy as np
# from EdgEnv import EdgEnv
from env import ArmEnv
# import parameters as param
from A2C import Agent

# 训练参数
MAX_EPISODES = 500
MAX_EP_STEPS = 200
# ON_TRAIN = True
LR_A = 0.001                           # actor 的学习率
LR_C = 0.01                            # critic 的学习率



# 设置多智能体环境
# env = EdgEnv()
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim

N_F = s_dim
N_A = a_dim


# 设置智能体
sess = tf.Session()
sess1 = tf.Session()
agent = Agent(0,sess,n_features=N_F,n_actions=N_A,lr_a=LR_A,lr_c=LR_C)
# agent1 = Agent(1,sess1,n_features=N_F,n_actions=N_A,lr_a=LR_A,lr_c=LR_C)
sess.run(tf.global_variables_initializer())
# sess1.run(tf.global_variables_initializer())


res = []

MAX_EPISODE = 500
RENDER = True
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = agent.actor.choose_action(s)

        # s_, r, done, info = env.step(a)
        s_, r, done = env.step(a)

        # if done: r = -20
        if done: 
            r = 0

        track_r.append(r)

        td_error = agent.critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        agent.actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

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





