import math

train_length = 8  # todo do we need this?
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

# ----------------Evolution---------------------

number_of_offspring = 5
# interpolation calculated by solving   10 * e^(fit + x) * y = 10 and 2
#                                       100 * e^(-fit + x) * y = 100 and 500


def population_size(fitness):
    x = 1 / 4 * (1 - 5 * math.exp(1))
    y = -(4 / (5 * (math.exp(1) - 1)))
    return math.floor(10 * (math.exp(fitness) + x) * y)  # interpolated form 10 to 2 note: 9 most of the time


def eval_rounds(fitness):
    x = (1 - 5 * math.exp(1)) / (4 * math.exp(1))
    y = -(4 * math.exp(1)) / (math.exp(1) - 1)
    return math.floor(100*(math.exp(-fitness) + x) * y)  # interpolated from 100 to 500
