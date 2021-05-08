# TF1.8.0

import numpy as np
import tensorflow as tf

GAMMA = 0.9                     # TD差奖励折扣


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):          # n_features为状态数
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")   # 状态，数据类型为tf.float32,数据形状为[1,n_features],名称为state
        self.a = tf.placeholder(tf.int32, None, "action")               # 动作,数据形状为一个整形
        self.td_error = tf.placeholder(tf.float32, None, "td_error")    # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,                                          # 输入
                units=20,                                               # 隐藏单元数
                activation=tf.nn.relu,                                  # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),# 权重
                bias_initializer=tf.constant_initializer(0.1),          # 偏置
                name='l1'                                               # 名称
            )

            self.acts_prob = tf.layers.dense(                           # 动作概率分布
                inputs=l1,                                              # 输入为l1
                units=n_actions,                                        # 输出单元个数
                activation=tf.nn.softmax,                               # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),# 权重
                bias_initializer=tf.constant_initializer(0.1),          # 偏置
                name='acts_prob'                                        # 名称
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            print("-------------")
            print("self.acts_prob",self.acts_prob)
            print("self.acts_prob[0,0]",self.acts_prob[0,0])
            print("self.acts_prob[0,1]",self.acts_prob[0,1])
            print("-------------")
            
            # 添加熵的计算？？？？
            log_prob1 = tf.log(self.acts_prob[0,:])
            Entropy = 0
            for i in range(n_actions):
                Entropy += tf.log(self.acts_prob[0,i]) * self.acts_prob[0,i]
            beta = 0.009
            self.exp_v = tf.reduce_mean(log_prob * self.td_error - beta * Entropy)
            
            # self.exp_v = tf.reduce_mean(log_prob * self.td_error)             # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}                 # 在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})                    # 返回最可能的动作
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # 动作用整数来表示


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")         # 状态
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")                # 预测值
        self.r = tf.placeholder(tf.float32, None, 'r')                        # 奖励值

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,                                                     # 隐藏单元数
                activation=tf.nn.relu,                                        # 激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),      # 权重
                bias_initializer=tf.constant_initializer(0.1),                # 偏置
                name='l1'                                                     # 名称
            )

            self.v = tf.layers.dense(
                inputs=l1,                                                    # 
                units=1,                                                      # 输出单元个数
                activation=None,                                              # 无激活函数
                kernel_initializer=tf.random_normal_initializer(0., .1),      # 权重
                bias_initializer=tf.constant_initializer(0.1),                # 偏置
                name='V'                                                      # 名称
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)                              # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error



class Agent(object):

    def __init__(self,ag_id,sess,n_features,n_actions,lr_a=0.001,lr_c=0.01):
        self.actor = Actor(sess,n_features,n_actions,lr=lr_a)
        self.critic = Critic(sess,n_features,lr=lr_c)
        self.ag_id = ag_id

    def getid(self):
        return self.ag_id

