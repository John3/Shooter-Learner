import numpy as np
from itertools import tee
import os


class LogEntry:
    def __init__(self, player, player_name, action, feature_vector, damage_arr, reward):
        self.player = player
        self.player_name = player_name
        self.action = int(action)
        self.feature_vector = feature_vector
        self.damage_arr = damage_arr
        self.reward = reward
        self.end = False

    def toArray(self):
        res = list()
        res.append(self.player)
        res.append(self.action)
        res.extend(self.feature_vector)
        res.extend(self.damage_arr)
        res.append(self.reward)
        res.append(self.end)
        return np.array(res)

    def __str__(self):
        return "Entry: (p: %d, pn: %s, a: %d, fv: %s, da: %s, r: %s, e %s)" % \
               (self.player, self.player_name, self.action, self.feature_vector, self.damage_arr, self.reward, self.end)


class LogFile:
    def __init__(self, log_entries):
        self.log_entries = log_entries
        self.index = 0

    def get_batch(self, size):
        batch = self.log_entries[self.index:size]
        if not batch:
            return False

        self.index += size
        actions = list()
        fvs = list()
        reward_sum = 0
        for entry in batch:
            actions.append(entry.action)
            fvs.append(entry.feature_vector)
            reward_sum += entry.reward

        return actions, fvs, reward_sum

    def __iter__(self):
        return iter(self.log_entries)

    def __getitem__(self, item):
        return self.log_entries[item]

    def __len__(self):
        return len(self.log_entries)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def action_value_map(x):
    return {
        'None': 0.,
        'MoveForward': 1.,
        'MoveLeft': 2.,
        'MoveRight': 3.,
        'MoveBackward': 4.,
        'TurnLeft': 5.,
        'TurnRight': 6.,
        'Shoot': 7.,
        'Prepare': 8.
    }[x]


def parse(filename):
    output = []

    with open(filename) as f:
        names = next(f)
        players = names.split('|')
        player1Name = players[0].split(':')[1]
        player2Name = players[1].split(':')[1]

        for line, next_line in pairwise(f.read().splitlines()):
            if line[0] == 'D':
                continue

            line_split = line.split(':')
            next_line_split = next_line.split(':')
            log_type = next_line_split[0]
            damage_arr = list()
            player_damage = -1.
            damage = 0.
            player_health = -1.
            reward = 0.
            if log_type == "D":
                player_damage = float(next_line_split[1])
                damage = float(next_line_split[2])
                player_health = float(next_line_split[3])

            if log_type == "R":
                reward = float(next_line_split[1])
            player = float(line_split[1])

            if player == 0:
                player_name = player1Name
            else:
                player_name = player2Name
            action = action_value_map(line_split[3])
            feature_vector = list(map(lambda x: float(x), line_split[2].split(' ')))

            damage_arr.append(player_damage)
            damage_arr.append(damage)
            damage_arr.append(player_health)
            if line_split[0] == "D" or line_split[0] == "R":
                continue
            output.append(LogEntry(player, player_name, action,
                                   feature_vector, damage_arr, reward))
    output[-1].end = True
    return LogFile(output)


def parse_logs_in_folder(folder_name):
    arrays = []
    for subdir, dirs, files in os.walk(folder_name):
        for file in files:
            arrays.append(parse(subdir + "/" +file))
    return arrays

#arrays = []
#for subdir, dirs, files in os.walk("/home/baljenurface/Speciality/Tensorflow/Gamelogs"):
#    for file in files:
#        arrays.append(parse(subdir + "/" +file))
#print(arrays)
#np.savez_compressed("compressed", arrays)
