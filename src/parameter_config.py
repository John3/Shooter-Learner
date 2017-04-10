import math

# ------------ Training parameters --------------

run_name = "FreksenThinkDeep"
batch_size = 4  # Number of traces to use for each training step
trace_length = 76  # How long each experience trace will be
discount_factor = .99

train_freq = 4
# How often do we train

load_model = True

save_path = "./dqn"
player_number = 0


def result_reward(winner):
    reward = 0
    if winner.startswith("player0"):
        reward = 1
    return reward


def meta_reward(last_action, last_enemy_health, fv):
    enemy_health = fv[10]
    reward = 0
    if last_action == 7 and enemy_health < last_enemy_health:
        reward = 0.00001
    return reward

rew_funcs = {
    "result_reward": result_reward,
    "meta_rewards": meta_reward
}

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

# ------------------Training--------------------
start_e = 1  # Starting probability of choosing a random action
end_e = 0.1  # Ending probability of choosing a random action
steps_e = 20000  # How many steps untill the probability of choosing a random action becomes end_e

step_drop = (start_e - end_e) / steps_e

buffer_size = 50000

# -----------------DDQRN Trainer----------------
pre_train_steps = 15000

fv_size = 15  # Size of the FeatureVector (state)


# ----------------------DDQRN-------------------
use_act = True
act_max_computation=10
tau = 0.001 # Rate to update target network toward primary network

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
