import numpy as np
import tensorflow as tf
import os
import parameter_config as cfg

from experience_buffer import ExperienceBuffer


class DDQRNTrainer:

    def __init__(self, ddqrn, target, sess):
        self.ddqrn = ddqrn
        self.target = target
        self.sess = sess
        self.batch_size = cfg.batch_size
        self.trace_length = cfg.trace_length
        self.buffer = ExperienceBuffer()

        self.j_list = []
        self.r_list = []

        self.total_steps = 0

        self.tensorboard_setup()
        self.summary = None

    def start_episode(self):
        self.episode_buffer = ExperienceBuffer()
        self.j = 0 # The number of steps we have taken
        self.r_all = 0

    def end_episode(self):
        self.buffer.add(self.episode_buffer.buffer)
        self.j_list.append(self.j)
        self.r_list.append(self.r_all)

        if self.summary is not None:
            train_count = self.ddqrn.sess.run([self.ddqrn.train_count])[0]
            if train_count % 10 == 0:
                self.train_writer.add_summary(self.summary, train_count)

    def experience(self, s, a, r, s1, end):
        if r != 0:
            print("Experienced a reward of: %s" % r)
        self.j += 1 # Increment the number of steps by one
        self.total_steps += 1
        # todo do we need this when we have it in the log_file? Maybe the log_file should be saved to an experience_buffer
        # Save the experience
        self.episode_buffer.add(np.reshape(np.array([s, a, r, s1, end]), [1, 5]))

        if self.total_steps > cfg.pre_train_steps:  # Only start training after some amount of steps, so we have something in our experience buffer

            if self.total_steps % cfg.train_freq == 0:

                # Reset the hidden state
                state_train = (np.zeros([self.batch_size, cfg.fv_size]), np.zeros([self.batch_size, cfg.fv_size]))

                train_batch = self.buffer.sample(self.batch_size, self.trace_length)  # Get a random batch of experiences

                # Perform the Double-DQN update to the target Q-Values
                # todo Lukas confused: Hvorfor skal de kun have "næste state" med?
                Q1 = self.ddqrn.get_prediction(
                    np.vstack(train_batch[:, 3]),
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
                target_Q = train_batch[:, 2] + (cfg.discount_factor * double_Q * end_multiplier)

                # Update the network with the target values

                _, self.summary = self.ddqrn.get_update_model(
                    self.merged,
                    np.vstack(train_batch[:, 0]),
                    target_Q=target_Q,
                    actions=train_batch[:, 1],
                    batch_size=self.batch_size,
                    train_length=self.trace_length,
                    state_in=state_train
                )

            if self.total_steps % 1000 == 0:
                self.target.update(self.sess)  # Set the target network to be equal to the primary network

        self.r_all += r


    def tensorboard_setup(self):
        self.merged = tf.summary.merge_all(self.ddqrn.scope)

        self.train_writer = tf.summary.FileWriter("summaries/logs/train/"+ cfg.run_name, self.ddqrn.sess.graph)
        self.test_writer = tf.summary.FileWriter("summaries/logs/test/" + cfg.run_name)
