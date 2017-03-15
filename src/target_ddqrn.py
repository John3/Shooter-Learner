class target_ddqrn:

    tau = 0.001 # Rate to update target network toward primary network

    def __init__(self, ddqrn, trainables):
        self.ddqrn = ddqrn
        self.target_ops = self.update_target_graph(trainables, self.tau)
        self.Q_out = ddqrn.Q_out

    def update(self, sess):
        # Sets the target network to be equal to the primary network
        self.update_target(self.target_ops, sess)

    # Functions for updating the target network todo Needs review (copy-pasta)
    def update_target_graph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        middle = total_vars // 2  # "Floor division"
        for idx, var in enumerate(tfVars[0:middle]):
            op_holder.append(
                tfVars[idx + middle].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + middle].value())))
        return op_holder

    # Couldn't this simply assign the target to the primary network? I.e. copy all the weights
    def update_target(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)
    # End todo