import tensorflow as tf

class DDQRN:

    def __init__(self, input_frames, train_length, input_size, output_size):
        self.train_length = train_length
        self.output_size = output_size
        self.input_size = input_size

        input_layer_output = self.build_input_layer(input_frames)
        lstm_layer_output = self.build_lstm_layer(input_layer_output)
        forward_layer_output = self.build_forward_layer(lstm_layer_output)
        self.output = self.build_output_layer(forward_layer_output)

    def build_input_layer(self, layer_input):
        return layer_input

    def build_lstm_layer(self, layer_input):
        return layer_input

    def build_forward_layer(self, layer_input):

        input_shape_size = layer_input.get_shape().as_list()[1]

        self.forward_W1 = tf.Variable(tf.random_normal([input_shape_size, input_shape_size]))
        self.forward_b1 = tf.Variable(tf.random_normal([input_shape_size]))

        return tf.tanh(tf.nn.xw_plus_b(layer_input, self.forward_W1, self.forward_b1))

    def build_output_layer(self, layer_input):

        input_shape_size = layer_input.get_shape().as_list()[1]

        self.output_W1 = tf.Variable(tf.random_normal([input_shape_size, self.output_size]))
        self.output_b1 = tf.Variable(tf.random_normal([self.output_size]))

        return tf.tanh(tf.nn.xw_plus_b(layer_input, self.output_W1, self.output_b1))