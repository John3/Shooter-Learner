import tensorflow as tf
from simple_ddqrn import DDQRN
from gameLogs.LogParser import parse_logs_in_folder

n_batches = 10
train_length = 8
fv_size = 15
actions_size = 8

input_frames = tf.placeholder(shape=[None, fv_size], dtype=tf.float32)


ddqrn = DDQRN(input_frames, train_length, fv_size, actions_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

logs = parse_logs_in_folder("data/game_logs")

for log_file in logs:
    for event in log_file:
        if event[0] == 0:
            output = sess.run([ddqrn.output], feed_dict={input_frames:[event[2:-4]]})

#for i in range(10):
#    output = sess.run([ddqrn.output], feed_dict={input_frames:})