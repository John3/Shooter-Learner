import tensorflow as tf
import numpy as np

class DDQRN:

    def __init__(self, sess, fv_size, input_size, output_size, scope):
        self.sess = sess
        self.output_size = output_size
        self.input_size = input_size
        self.scope = scope

        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32)

        self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.input_frames = tf.placeholder(shape=[None, fv_size], dtype=tf.float32)

        self.cell = tf.contrib.rnn.LSTMCell(num_units=input_size, state_is_tuple=True)
        self.state = (np.zeros([1, fv_size]), np.zeros([1, fv_size]))

        input_layer_output = self.build_input_layer(self.input_frames)
        lstm_layer_output = self.build_lstm_layer(input_layer_output)
        forward_layer_output = self.build_forward_layer(lstm_layer_output)
        self.predict, self.update_model, self.Q_out = self.build_output_layer(forward_layer_output)


    def build_input_layer(self, layer_input):
        return layer_input

    def build_lstm_layer(self, layer_input):
        input_shape_size = layer_input.get_shape().as_list()[1]

        inputFlat = tf.reshape(tf.contrib.layers.flatten(layer_input), [self.batch_size, self.train_length, input_shape_size])

        self.state_in = self.cell.zero_state(self.batch_size, tf.float32)
        rnn, self.rnn_state = tf.nn.dynamic_rnn(
            inputs=inputFlat,
            cell=self.cell,
            dtype=tf.float32,
            initial_state=self.state_in,
            scope=self.scope
        )
        rnn = tf.reshape(rnn, shape=[-1, input_shape_size])

        return rnn

    def build_forward_layer(self, layer_input):

        input_shape_size = layer_input.get_shape().as_list()[1]

        forward_size_1 = self.output_size * 15

        forward_W1 = tf.Variable(tf.random_normal([input_shape_size, forward_size_1]))
        forward_b1 = tf.Variable(tf.random_normal([forward_size_1]))

        forward_1 = tf.tanh(tf.nn.xw_plus_b(layer_input, forward_W1, forward_b1))

        forward_size_2 = self.output_size * 5

        forward_W2 = tf.Variable(tf.random_normal([forward_size_1, forward_size_2]))
        forward_b2 = tf.Variable(tf.random_normal([forward_size_2]))

        forward_2 = tf.tanh(tf.nn.xw_plus_b(forward_1, forward_W2, forward_b2))

        return forward_2

    def build_output_layer(self, layer_input):

        input_shape_size = layer_input.get_shape().as_list()[1]

        # Split the input into value and advantage
        streamA, streamV = (layer_input, layer_input) #tf.split(self.output_1, 2, 1)
        #self.streamA = tf.contrib.layers.flatten(self.streamAC)
        #self.streamV = tf.contrib.layers.flatten(self.streamVC)

        AW = tf.Variable(tf.random_normal([input_shape_size, self.output_size]))
        VW = tf.Variable(tf.random_normal([input_shape_size, 1]))

        advantage = tf.matmul(streamA, AW)
        value = tf.matmul(streamV, VW)

        # Combine value and advantage to get our Q-values
        Q_out = value + (advantage - tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True))
        predict = tf.argmax(Q_out, 1)

        # We can obtain the loss when we are given the target Q value by taking the SSD between target and prediction
        actions_onehot = tf.one_hot(self.actions, self.output_size, dtype=tf.float32)

        Q = tf.reduce_sum(Q_out * actions_onehot, reduction_indices=1)

        td_error = tf.square(self.target_Q - Q)
        loss = tf.reduce_mean(td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        update_model = self.trainer.minimize(loss)

        return predict, update_model, Q_out

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

    def get_Q_out(self, input, train_length, batch_size, state_in):
        return self.sess.run(self.Q_out, feed_dict={
            self.input_frames: input,
            self.train_length: train_length,
            self.batch_size: batch_size,
            self.state_in: state_in
        })

    def get_update_model(self, input, target_Q, actions, train_length, batch_size, state_in):
        self.sess.run(self.update_model, feed_dict={
            self.input_frames: input,
            self.target_Q: target_Q,
            self.actions: actions,
            self.train_length: train_length,
            self.batch_size: batch_size,
            self.state_in: state_in
        })