import tensorflow as tf

class DDQRN:

    def __init__(self, input_frames, train_length, input_size, output_size):
        self.train_length = train_length
        self.output_size = output_size
        self.input_size = input_size

        self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        input_layer_output = self.build_input_layer(input_frames)
        lstm_layer_output = self.build_lstm_layer(input_layer_output)
        forward_layer_output = self.build_forward_layer(lstm_layer_output)
        self.predict, self.update_model, self.Q_out = self.build_output_layer(forward_layer_output)


    def build_input_layer(self, layer_input):
        return layer_input

    def build_lstm_layer(self, layer_input):
        return layer_input

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