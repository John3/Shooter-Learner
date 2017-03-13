import numpy as np
from itertools import tee
import os


class LogEntry:
    def __init__(self, player, playerName, action, featureVector, damageArr, reward):
        self.player = player
        self.playerName = playerName
        self.action = action
        self.featureVector = featureVector
        self.damageArr = damageArr
        self.reward = reward

    def toArray(self):
        res = list()
        res.append(self.player)
        res.append(self.action)
        res.extend(self.featureVector)
        res.extend(self.damageArr)
        res.append(self.reward)
        return np.array(res)


class LogFile:
    def __init__(self, log_entries):
        self.log_entries = log_entries
        self.index = 0

    def getBatch(self, size):
        batch = self.log_entries[self.index:size]
        if not batch:
            return False

        self.index += size
        actions = list()
        fvs = list()
        rewardSum = 0
        for entry in batch:
            actions.append(entry.action)
            fvs.append(entry.featureVector)
            rewardSum += entry.reward

        return actions, fvs, rewardSum


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
            next_line_split = next_line.split(':')
            log_type = next_line_split[0]
            damage_arr = list()
            playerDamage = -1.
            damage = 0.
            playerhealth = -1.
            reward = 0.
            if log_type == "D":
                playerDamage = float(next_line_split[1])
                damage = float(next_line_split[2])
                playerhealth = float(next_line_split[3])

            if log_type == "R":
                reward = float(next_line_split[1])
            player = float(line_split[1])

            if player == 0:
                playername = player1Name
            else:
                playername = player2Name
            action = action_value_map(line_split[3])
            featureVector = map(lambda x: float(x), line_split[2].split(' '))

            damage_arr.append(playerDamage)
            damage_arr.append(damage)
            damage_arr.append(playerhealth)
            line_split = line.split(':')
            if line_split[0] == "D" or line_split[0] == "R":
                continue
            output.append(LogEntry(player, playername, action,
                                   featureVector, damage_arr, reward))
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
