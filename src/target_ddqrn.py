import tensorflow as tf
import parameter_config as cfg


class target_ddqrn:
    def __init__(self, ddqrn, trainables):
        self.ddqrn = ddqrn
        self.tau = tf.placeholder(tf.float32)
        self.target_ops = self.update_target_graph(trainables)
        self.Q_out = ddqrn.Q_out

    def update(self, sess, tau=cfg.tau):
        # Sets the target network to be equal to the primary network
        self.update_target(self.target_ops, sess, tau=tau)

    def get_Q_out(self, input, train_length, batch_size, state_in):
        return self.ddqrn.get_Q_out(input, train_length, batch_size, state_in)

    # Functions for updating the target network todo Needs review (copy-pasta)
    def update_target_graph(self, tfVars):
        with tf.name_scope("update_target_graph"):
            op_holder = []
            for idx, var in enumerate(tfVars[0]):
                op_holder.append(
                    tfVars[1, idx].assign((var.value() * self.tau) + ((1 - self.tau) * tfVars[1, idx].value())))
            return op_holder

    # Couldn't this simply assign the target to the primary network? I.e. copy all the weights
    def update_target(self, op_holder, sess, tau=cfg.tau):
        for op in op_holder:
            sess.run(op, feed_dict={self.tau: tau})
    # End todo