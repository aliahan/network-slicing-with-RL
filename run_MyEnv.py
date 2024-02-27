
import gymnasium
from RL_brain import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from UREnv import MyEnv

tf.reset_default_graph()   # bastan va khali kardane hafezeye tf haye ghabli
MEMORY_SIZE = 100
ACTION_SPACE = 34
Numb_features=3
mode="WD"
D=10
E=10
#env=env_network(mode,D,E)
env = MyEnv(learning_windows=2000)
# env = env.unwrapped
# env.seed(1)

sess = tf.Session()
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=Numb_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=False)

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=Numb_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    acc_r = [0]
    action_list=[]

    total_steps = 0
    observation = env.reset()
    #print("observation",observation)
    while True:
        # if total_steps-MEMORY_SIZE > 9000: env.render()

        action = RL.choose_action(observation)
        action_list.append(action)
        # print("teeeeeeest",action_list)
        # print(action_list[0])
        # print(action_list[1])
        # print(action_list[2])
        # print("action",action)
        #f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
        # print("**********************************")
        # print("observation",observation)
        observation_, reward, done, info = env.step(np.array([action]))
        # print("observation_",observation_)
        acc_r.append(reward + acc_r[-1])  # accumulated reward
        # print("observation.shape,action,reward,observation_.shape")
        # print("#################")
        # print("observation",observation)
        # print("revar",reward)
        # print("observation_",observation_)
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()

        if done:
            break

        observation = observation_
        total_steps += 1
        print("step: ", total_steps)
    return RL.cost_his, acc_r , action_list

c_natural, r_natural , a_natural = train(natural_DQN)
print("chaaaaaangeeed")
c_dueling, r_dueling , a_dueling = train(dueling_DQN) 

plt.figure(1)
plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()