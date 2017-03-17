import numpy as np
import tensorflow as tf
import os

from experience_buffer import ExperienceBuffer


class DDQRNTrainer:

    pre_train_steps = 10000

    start_e = 1  # Starting probability of choosing a random action
    end_e = 0.1  # Ending probability of choosing a random action
    steps_e = 10000  # How many steps untill the probability of choosing a random action becomes end_e

    step_drop = (start_e - end_e) / steps_e

    train_freq = 4
    discount_factor = 0.99

    fv_size = 15  # Size of the FeatureVector (state)

    def __init__(self, ddqrn, target, sess, batch_size, trace_length):
        self.ddqrn = ddqrn
        self.target = target
        self.sess = sess
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.buffer = ExperienceBuffer()

        self.j_list = []
        self.r_list = []

        self.total_steps = 0
        self.e = 0
        # Set the rate of random action decrease.
        self.e = self.start_e

        self.saver = tf.train.Saver()


    def start_episode(self):
        self.episode_buffer = ExperienceBuffer()
        self.j = 0 # The number of steps we have taken
        self.r_all = 0



    def end_episode(self):
        self.buffer.add(self.episode_buffer.buffer)
        self.j_list.append(self.j)
        self.r_list.append(self.r_all)


    def experience(self, s, a, r, s1, end):
        self.j += 1 # Increment the number of steps by one
        self.total_steps += 1

        # todo do we need this when we have it in the log_file? Maybe the log_file should be saved to an experience_buffer
        # Save the experience
        self.episode_buffer.add(np.reshape(np.array([s, a, r, s1, end]), [1, 5]))

        if self.total_steps > self.pre_train_steps:  # Only start training after some amount of steps, so we have something in our experience buffer
            if self.e > self.end_e:
                self.e -= self.step_drop

            if self.total_steps % self.train_freq == 0:

                # Reset the hidden state
                state_train = (np.zeros([self.batch_size, self.fv_size]), np.zeros([self.batch_size, self.fv_size]))

                train_batch = self.buffer.sample(self.batch_size, self.trace_length)  # Get a random batch of experiences

                # Perform the Double-DQN update to the target Q-Values
                # todo Lukas confused: Hvorfor skal de kun have "næste state" med?
                Q1 = self.ddqrn.get_prediction(
                    np.vstack(train_batch[:, 3]), # todo Divide by 255?
                    batch_size=self.batch_size,
                    train_length=self.trace_length,
                    state_in=state_train
                )

                Q2 = self.target.get_Q_out(
                    np.vstack(train_batch[:, 3]),
                    batch_size=self.batch_size,
                    train_length=self.trace_length,
                    state_in=state_train
                )

                end_multiplier = -(train_batch[:, 4] - 1)
                double_Q = Q2[range(self.batch_size * self.trace_length), Q1]
                target_Q = train_batch[:, 2] + (self.discount_factor * double_Q * end_multiplier)

                # Update the network with the target values
                _ = self.ddqrn.get_update_model(
                    np.vstack(train_batch[:, 0]),
                    target_Q=target_Q,
                    actions=train_batch[:, 1],
                    batch_size=self.batch_size,
                    train_length=self.trace_length,
                    state_in=state_train
                )

                if self.total_steps % 1000 == 0:
                    self.target.update(self.sess)  # Set the target network to be equal to the primary network

        self.target.update(self.sess)  # Set the target network to be equal to the primary network
        self.r_all += r


    def load(self, path):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save(self, path):
        # Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)

        self.saver.save(self.sess, path + '/model')