import tensorflow as tf
import numpy as np
import random
import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
tf.reset_default_graph()
memory_capacity = 50
state_size = 16
memory = tf.placeholder(shape=[state_size, memory_capacity, state_size], dtype=tf.float32)
cell = tf.contrib.rnn.GRUCell(state_size)
initial_state = tf.zeros([state_size, state_size])
outputs, state = tf.nn.dynamic_rnn(cell, memory, initial_state=initial_state)
W = tf.Variable(tf.random_uniform([state_size, 4], 0, 0.01))
output = tf.reshape(outputs, shape=[-1, state_size])
Qout = tf.matmul(output, W)
predict = tf.argmax(Qout, 1)
nextQ = tf.placeholder(shape=[800, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()
y = 0.99
e = 0.1
num_episodes = 2000
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        memmorycell = i % memory_capacity
        s = env.reset()
        stateMemmory = np.zeros((state_size, memory_capacity, state_size))
        stateMemmory.put(memmorycell, s)
        rAll = 0
        d = False
        j = 0
        while j < 99:
            j += 1
            a, allQ = sess.run([predict, Qout], feed_dict={memory: stateMemmory})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            s1, r, d, _ = env.step(a[0])
            memmorycell1 = memmorycell + 1 % memory_capacity
            stateMemmory.put(memmorycell1, s1)
            Q1 = sess.run(Qout, feed_dict={memory: stateMemmory})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1
            _, W1 = sess.run([updateModel, W], feed_dict={memory: stateMemmory, nextQ: targetQ})
            rAll += r
            s = s1
            if d == True:
                e = 1. / ((i / 50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%"
plt.figure()
plt.plot(jList)
plt.figure()
plt.plot(rList)
plt.show()
