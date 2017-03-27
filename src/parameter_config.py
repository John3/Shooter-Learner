train_length = 8  # todo do we need this?
fv_size = 15  # Size of the FeatureVector (state)
actions_size = 8  # Number of actions we can take
batch_size = 4  # Number of traces to use for each training step
trace_length = 8  # How long each experience trace will be
discount_factor = .99

train_freq = 4
# How often do we train