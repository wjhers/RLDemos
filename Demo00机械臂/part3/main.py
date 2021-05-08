# # # # 导入环境
from part3.env import ArmEnv
from part3.rl import DDPG
# 设置全局变量
MAX_EPISODES = 500
MAX_EP_STEPS = 200
# 设置环境
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# 设置学习方法
rl = DDPG(s_dim,a_dim,a_bound)

# 开始训练
for i in range(MAX_EPISODES):
    s = env.reset()
    for j in range(MAX_EP_STEPS):
        env.render()
        
        a = rl.choose_action(s)
        s_,r,done = env.step(a)
        
        # 存储记忆库
        rl.store_transition(s,a,r,s_)
        
        if rl.memory_full:
            rl.learn()
            
        s = s_

