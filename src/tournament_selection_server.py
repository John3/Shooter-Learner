import json
import os

from evolution_trainer import EvolutionHost, Individual
from typing import List
import random
import tensorflow as tf
import numpy as np
import parameter_config as cfg


class TournamentSelectionServer:

    def __init__(self, population: List[Individual], evaluation_rounds, saver: tf.train.Saver):
        self.saver = saver
        self.evaluation_rounds = evaluation_rounds
        self.population = population
        self.evaluated_population = []
        self.population_size = len(population)
        self.population.extend(self.generate_new_population())
        self.current_individual = self.population.pop()
        self.current_round = 0
        self.current_reward = 0
        self.rnn_state = None

    def generate_new_population(self) -> List[Individual]:
        new_population = []
        for ind in self.population:
            for x in range(self.population_size):
                new_population.append(ind.generate_offspring(x))
        return new_population

    def update_graph(self, individual):
        tensor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_DDQRN")
        for tenVar, var in zip(tensor_variables, individual.variables):
            tenVar.assign(var)

    def callback(self, msg):
        if msg["type"] == "instruction":
            return {"type": "instruction", "command": "resetMission"}

        if msg["type"] == "event":
            if msg["message"] == "game_start":
                # Reset state
                self.rnn_state = (np.zeros([cfg.batch_size, cfg.fv_size]), np.zeros([cfg.batch_size, cfg.fv_size]))
                print("Game has started!")
                return {"response": "success"}
            if msg["message"] == "game_end":
                self.current_round += 1
                print("Game ended with result: " + str(msg))
                winner = msg["result"]
                if winner.startswith("player0"):
                    self.current_reward += 1
                if self.current_round > self.evaluation_rounds:
                    self.current_individual.fitness = self.current_reward/self.evaluation_rounds
                    self.evaluated_population.append(self.current_individual)
                    self.current_round = 0
                    self.current_reward = 0
                    if len(self.population) == 0:
                        self.selection()
                    self.current_individual = self.population.pop()
                    self.update_graph(self.current_individual)
                return {"response": "success"}

        if msg["type"] == "think":
            fv1 = self.json_string_to_feature_vector(msg["feature_vector"])
            predict = tf.get_collection("predict", scope="main_DDQRN")[0]

            sess = tf.get_default_session()
            a, self.rnn_state = sess.run([predict, self.rnn_state], feed_dict={
                "input_frames:0": [fv1],
                "train_length:0": cfg.train_length,
                "batch_size:0": cfg.batch_size,
                "state_in:0": self.rnn_state
            })
            a = a[0].item()

            return a

        print("Unhandled message: " + str(msg))
        return 3

    @staticmethod
    def json_string_to_feature_vector(json_str):
        fv = json.loads(json_str)
        out = [
            fv["DeltaRot"],
            fv["DeltaMovedX"],
            fv["DeltaMovedY"],
            fv["VelX"],
            fv["VelY"],
            fv["DamageProb"],
            fv["DeltaDamageProb"],
            fv["DistanceToObstacleLeft"],
            fv["DistanceToObstacleRight"],
            fv["Health"],
            fv["EnemyHealth"],
            fv["TickCount"],
            fv["TicksSinceObservedEnemy"],
            fv["TicksSinceDamage"],
            fv["ShootDelay"]
        ]
        return out

    def selection(self) -> List[Individual]:
        self.population = []
        sample = self.random_sample(self.population_size)
        while sample:
            sample.sort(key=lambda x: x.fitness, reverse=True)
            self.population.append(sample[0])
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.update_graph(self.population[0])
        sess = tf.get_default_session()
        path = "./dqn"
        if not os.path.exists(path):
            os.makedirs(path)

        self.saver.save(sess, path + '/model-evolution')
        self.population.extend(self.generate_new_population())

    def random_sample(self, count) -> List[Individual]:
        res = []
        for k in range(count):
            i = random.randint(0, len(self.evaluated_population))
            if len(self.evaluated_population) > 0:
                res.append(self.evaluated_population.pop(i))
        return res

if __name__ == "__main__":
    host = EvolutionHost("./dqn/model", "host")
    population = [host.individual.generate_offspring(i) for i in range(10)]
    TournamentSelectionServer(population, 50, host.saver)
