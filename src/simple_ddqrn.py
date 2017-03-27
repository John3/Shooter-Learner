import tensorflow as tf
import numpy as np

from external.act_cell import ACTCell

class DDQRN:

    def __init__(self, sess, fv_size, input_size, output_size, scope):
        self.sess = sess
        self.output_size = output_size
        self.input_size = input_size
        self.scope = scope

        with tf.name_scope(scope):
            self.train_length = tf.placeholder(dtype=tf.int32, name="train_length")
            self.batch_size = tf.placeholder(dtype=tf.int32, name="batch_size")

            self.target_Q = tf.placeholder(name="target_Q", shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(name="actions", shape=[None], dtype=tf.int32)
            self.input_frames = tf.placeholder(name="input_frames", shape=[None, fv_size], dtype=tf.float32)

            self.cell = tf.contrib.rnn.LSTMCell(num_units=input_size)
            #self.cell = ACTCell(num_units=input_size, cell=self.inner_cell, epsilon=0.01,
            #                    max_computation=50, batch_size=self.batch_size)
            self.state = (np.zeros([1, fv_size]), np.zeros([1, fv_size]))

            input_layer_output = self.build_input_layer(self.input_frames)
            lstm_layer_output = self.build_lstm_layer(input_layer_output)
            forward_layer_output = self.build_forward_layer(lstm_layer_output)
            self.predict, Q, self.Q_out = self.build_output_layer(forward_layer_output)

            with tf.name_scope("loss"):
                td_error = tf.square(self.target_Q - Q)
                loss = tf.reduce_mean(td_error)
                tf.summary.scalar("loss", loss, [scope])
                tf.summary.scalar("Q", tf.reduce_mean(Q), [scope])
                tf.summary.scalar("target_Q", tf.reduce_mean(self.target_Q), [scope])

            with tf.name_scope("training"):
                self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
                self.update_model = self.trainer.minimize(loss)

            with tf.name_scope("counters"):
                self.evaluation_count = tf.Variable(0, dtype=tf.int32, trainable=False, name="evaluation_count")
                self.inc_evaluation_count = tf.assign_add(self.evaluation_count, 1, name="inc_evaluation_count")
                self.train_count = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_count")
                self.inc_train_count = tf.assign_add(self.train_count, 1, name="inc_train_count")

    def build_input_layer(self, layer_input):
        return layer_input

    def build_lstm_layer(self, layer_input):
        with tf.name_scope('lstm_layer'):
            input_shape_size = layer_input.get_shape().as_list()[1]

            with tf.name_scope('flatten_input'):
                inputFlat = tf.reshape(tf.contrib.layers.flatten(layer_input), [self.batch_size, self.train_length, input_shape_size])

            with tf.name_scope('build_lstm') as scope:
                self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
                rnn, self.rnn_state = tf.nn.dynamic_rnn(
                    inputs=inputFlat,
                    cell=self.cell,
                    dtype=tf.float32,
                    initial_state=self.state_in,
                    scope=scope
                )

            with tf.name_scope('reshape_output'):
                rnn = tf.reshape(rnn, shape=[-1, input_shape_size])

        return rnn

    def build_forward_layer(self, layer_input):
        with tf.name_scope('forward_layer'):
            input_shape_size = layer_input.get_shape().as_list()[1]
            keep_prob = tf.constant(0.5, dtype=tf.float32, name="keep_prob")

            with tf.name_scope("forward_layer_1"):
                forward_size_1 = self.output_size * 3

                forward_W1 = tf.Variable(tf.random_normal([input_shape_size, forward_size_1]), name="weights")
                forward_b1 = tf.Variable(tf.random_normal([forward_size_1]), name="biases")

                forward_1 = tf.tanh(tf.nn.xw_plus_b(layer_input, forward_W1, forward_b1), name="activation")

                forward_1 = tf.nn.dropout(forward_1, keep_prob)

            with tf.name_scope("forward_layer_2"):
                forward_size_2 = self.output_size * 2

                forward_W2 = tf.Variable(tf.random_normal([forward_size_1, forward_size_2]), name="weights")
                forward_b2 = tf.Variable(tf.random_normal([forward_size_2]), name="biases")

                forward_2 = tf.tanh(tf.nn.xw_plus_b(forward_1, forward_W2, forward_b2), name="activation")

                return tf.nn.dropout(forward_2, keep_prob)

    def build_output_layer(self, layer_input):
        with tf.name_scope('output_layer'):
            input_shape_size = layer_input.get_shape().as_list()[1]

            with tf.name_scope("advantage_value"):
                # Split the input into value and advantage
                streamA, streamV = (layer_input, layer_input)

                AW = tf.Variable(tf.random_normal([input_shape_size, self.output_size]), name="advantage_weights")
                VW = tf.Variable(tf.random_normal([input_shape_size, 1]), name="value_weights")

                advantage = tf.matmul(streamA, AW, name="advantage")
                value = tf.matmul(streamV, VW, name="value")

                # Combine value and advantage to get our Q-values
                Q_out = tf.add(value, (advantage - tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True)), name="Q_out")

            with tf.name_scope("prediction"):
                predict = tf.argmax(Q_out, 1)

                self.predict_summary = tf.summary.histogram("predictions", predict, [self.scope, "test"])

                # We can obtain the loss when we are given the target Q value by taking the SSD between target and prediction
                actions_onehot = tf.one_hot(self.actions, self.output_size, dtype=tf.float32)

                Q = tf.reduce_sum(Q_out * actions_onehot, reduction_indices=1)

        return predict, Q, Q_out

    def get_prediction(self, input, train_length, batch_size, state_in):
        return self.sess.run(self.predict, feed_dict={
            self.input_frames: input,
            self.train_length: train_length,
            self.batch_size: batch_size,
            self.state_in: state_in
        })

    def get_prediction_with_state(self, input, train_length, batch_size, state_in):
        return self.sess.run([self.predict, self.rnn_state], feed_dict={
            self.input_frames: input,
            self.train_length: train_length,
            self.batch_size: batch_size,
            self.state_in: state_in
        })

    def get_state(self, input, train_length, batch_size, state_in):
        return self.sess.run([self.rnn_state], feed_dict={
            self.input_frames: input,
            self.train_length: train_length,
            self.batch_size: batch_size,
            self.state_in: state_in
        })

    def get_Q_out(self, input, train_length, batch_size, state_in):
        return self.sess.run(self.Q_out, feed_dict={
            self.input_frames: input,
            self.train_length: train_length,
            self.batch_size: batch_size,
            self.state_in: state_in
        })

    def get_update_model(self, merged, input, target_Q, actions, train_length, batch_size, state_in):
        return self.sess.run([self.update_model, merged], feed_dict={
            self.input_frames: input,
            self.target_Q: target_Q,
            self.actions: actions,
            self.train_length: train_length,
            self.batch_size: batch_size,
            self.state_in: state_in
        })