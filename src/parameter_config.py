train_length = 8  # todo do we need this?
fv_size = 15  # Size of the FeatureVector (state)
actions_size = 8  # Number of actions we can take
batch_size = 4  # Number of traces to use for each training step
trace_length = 8  # How long each experience trace will be
discount_factor = .99

train_freq = 4
# How often do we train

action_to_string = {
    0: "none",
    1: "moveForward",
    2: "moveLeft",
    3: "moveRight",
    4: "moveBackward",
    5: "turnLeft",
    6: "turnRight",
    7: "shoot",
    8: "prepare"
}

features = [
    "DeltaRot",
    "DeltaMovedX",
    "DeltaMovedY",
    "VelX",
    "VelY",
    "DamageProb",
    "DeltaDamageProb",
    "DistanceToObstacleLeft",
    "DistanceToObstacleRight",
    "Health",
    "EnemyHealth",
    "TickCount",
    "TicksSinceObservedEnemy",
    "TicksSinceDamage",
    "ShootDelay",
]

prediction_to_action = [0, 1, 2, 3, 4, 5, 6, 7, 8]