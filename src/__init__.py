import tensorflow as tf
import numpy as np
import os

from experience_buffer import experience_buffer
from log_parser import parse_logs_in_folder
from sharpshooter_server import SharpShooterServer
from simple_ddqrn import DDQRN


# Functions for updating the target network todo Needs review (copy-pasta)
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    middle = total_vars // 2 # "Floor division"
    for idx, var in enumerate(tfVars[0:middle]):
        op_holder.append(tfVars[idx+middle].assign((var.value()*tau) + ((1-tau)*tfVars[idx+middle].value())))
    return op_holder

# Couldn't this simply assign the target to the primary network? I.e. copy all the weights
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
# End todo

train_length = 8 #todo do we need this?
fv_size = 15 # Size of the FeatureVector (state)
actions_size = 8 # Number of actions we can take
batch_size = 32 # Number of experiences to use for each training step
discount_factor = .99

train_freq = 4 # How often do we train

start_e = 1 # Starting probability of choosing a random action
end_e = 1 # Ending probability of choosing a random action
steps_e = 10000 # How many steps untill the probability of choosing a random action becomes end_e

tau = 0.001 # Rate to update target network toward primary network

# Set the rate of random action decrease.
e = start_e
step_drop = (start_e - end_e)/steps_e

input_frames = tf.placeholder(shape=[None, fv_size], dtype=tf.float32)

ddqrn = DDQRN(input_frames, train_length, fv_size, actions_size)
ddqrn_target = DDQRN(input_frames, train_length, fv_size, actions_size)

trainables = tf.trainable_variables()
target_ops = updateTargetGraph(trainables,tau)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Sets the target network to be equal to the primary network
updateTarget(target_ops, sess)

logs = parse_logs_in_folder("data/game_logs")

total_steps = 0
pre_train_steps = 10000 #How many steps of random actions before training begins.
# todo do we want to take many random actions before training? We can build the experience_buffer from the log_files

buffer = experience_buffer()

j_list = []
r_list = []

saver = tf.train.Saver()
load_model = False
save_path = "./dqn"

#Make a path for our model to be saved in.
if not os.path.exists(save_path):
    os.makedirs(save_path)

if load_model == True:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(save_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
for log_file in logs:
    episode_buffer = experience_buffer()
    j = 0 # The number of steps we have taken
    r_all = 0
    for i, event in enumerate(log_file):

        j += 1 # Increment the number of steps by one

        # Observe play
        s = event.feature_vector
        a = event.action
        s1 = log_file[i + 1].feature_vector
        r = log_file[i + 1].reward
        end = log_file[i + 1].end

        if event.player == 1:
            if end:
                break
            else:
                continue

        total_steps += 1

        # todo do we need this when we have it in the log_file? Maybe the log_file should be saved to an experience_buffer
        # Save the experience
        episode_buffer.add(np.reshape(np.array([s, a, r, s1, end]), [1, 5]))

        if total_steps > pre_train_steps: # Only start training after some amount of steps, so we have something in our experience buffer
            if e > end_e:
                e -= step_drop

            if total_steps % train_freq == 0:
                train_batch = buffer.sample(batch_size) # Get a random batch of experiences

                # Perform the Double-DQN update to the target Q-Values
                #todo Lukas confused: Hvorfor skal de kun have "næste state" med?
                Q1 = sess.run(ddqrn.predict, feed_dict={input_frames:np.vstack(train_batch[:, 3])})

                Q2 = sess.run(ddqrn_target.Q_out, feed_dict={input_frames:np.vstack(train_batch[:, 3])})

                end_multiplier = -(train_batch[:, 4] - 1)
                double_Q = Q2[range(batch_size), Q1]
                target_Q = train_batch[:, 2] + (discount_factor * double_Q * end_multiplier)

                # Update the network with the target values
                _ = sess.run(ddqrn.update_model, feed_dict={
                    input_frames: np.vstack(train_batch[:, 0]),
                    ddqrn.target_Q: target_Q,
                    ddqrn.actions: train_batch[:, 1]
                })

                updateTarget(target_ops, sess) # Set the target network to be equal to the primary network

        r_all += r

        if end:
            break
    buffer.add(episode_buffer.buffer)
    j_list.append(j)
    r_list.append(r_all)

saver.save(sess,save_path+'/model-'+str(len(logs))+'.cptk')
print("Done training!")

# Assuming we have now done some kind of training.. Try to predict some actions!


def server_cb(msg):
    print("Received message: %s" % str(msg))

    if msg["type"] == "instruction":
        return {"type": "instruction", "command": "resetMission"}

    if msg["type"] == "event":
        if msg["message"] == "game_start":
            print("Game has started!")
            return {"response": "success"}
        if msg["message"] == "game_end":
            print("Game has ended!")
            return {"response": "success"}
    if msg["type"] == "result":
        print("Game ended with result: " + str(msg))
        return {"response": "success"}

    if msg["type"] == "think":
        a = sess.run(ddqrn.predict, feed_dict={input_frames: [msg.feature_vector]})[0]
        return a

    print("Unhandled message: " + str(msg))
    return 3

server = SharpShooterServer(server_cb)
server.start()
while True:
    server.receive_message()

#for i in range(10):
#    output = sess.run([ddqrn.output], feed_dict={input_frames:})