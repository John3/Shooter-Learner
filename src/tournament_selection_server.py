import json
import os

from evolution_trainer import  Individual
from typing import List
import random
import tensorflow as tf
import numpy as np
import parameter_config as cfg


class TournamentSelectionServer:

    def __init__(self, ddqrn, population: List[Individual],
                 saver: tf.train.Saver, writer, reward_functions):
        self.ddqrn = ddqrn
        self.saver = saver
        self.writer = writer
        self.evaluation_rounds = cfg.eval_rounds(0)
        self.reward_functions = reward_functions
        self.population = population
        self.evaluated_population = []
        self.base_population_size = cfg.population_size(0)
        self.number_of_offspring = cfg.number_of_offspring
        self.population.extend(self.generate_new_population())
        self.current_individual = self.population.pop()
        self.current_round = 0
        self.current_reward = 0
        self.rnn_state = None
        self.game_has_ended = False

        self.last_enemy_health = 20
        self.a = None

        tensor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_DDQRN")
        self.variable_value = tf.placeholder(tf.float32)
        self.variable_update_ops = []
        for var in tensor_variables:
            self.variable_update_ops.append(var.assign(self.variable_value))

        self.fitness_tensor = tf.placeholder(tf.float32)
        self.fitness_summary = tf.summary.scalar("evolution/fitness", self.fitness_tensor)

    def save_population(self):
        if not os.path.exists("data/"):
            os.makedirs("data/")

        np.savez_compressed("data/evolution", population=self.population)

    def load_population(self):
        with np.load("data/evolution.npz") as data:
            self.population = list(data['population'])
        self.current_individual = self.population.pop()

    def generate_new_population(self) -> List[Individual]:
        new_population = []
        for ind in self.population:
            for x in range(self.number_of_offspring):
                new_population.append(ind.generate_offspring(x))
        return new_population

    def update_graph(self, individual):
        for tenVar, var in zip(self.variable_update_ops, individual.variables):
            self.ddqrn.sess.run([tenVar], feed_dict={self.variable_value: var})

    def callback(self, msg):
        self.game_has_ended = False
        if msg["type"] == "instruction":
            return {"type": "instruction", "command": "resetMission"}

        if msg["type"] == "event":
            if msg["message"] == "game_start":
                # Reset state
                self.rnn_state = (np.zeros([1, len(cfg.features)]), np.zeros([1, len(cfg.features)]))
                print("Game has started!")
                return {"response": "success"}
            if msg["message"] == "game_end":
                self.game_has_ended = True
                self.current_round += 1
                print("Game ended with result: " + str(msg))

                self.current_reward += self.reward_functions["result_reward"](msg["result"])
                if self.current_round == self.evaluation_rounds:
                    self.current_individual.fitness = self.current_reward/self.evaluation_rounds
                    print("Finished evaluating an individual, fitness was %s" % self.current_individual.fitness)
                    self.evaluated_population.append(self.current_individual)
                    self.current_round = 0
                    self.current_reward = 0
                    print("There are %s individuals left" % len(self.population))
                    if len(self.population) == 0:
                        print("Selecting an individual")
                        self.selection()
                    self.current_individual = self.population.pop()
                    self.update_graph(self.current_individual)
                return {"response": "success"}

        if msg["type"] == "think":
            fv1 = self.json_string_to_feature_vector(msg["feature_vector"])

            enemy_health = fv1[10]

            r = self.reward_functions["meta_rewards"](self.a, self.last_enemy_health, fv1)
            self.current_reward += r

            a, self.rnn_state = self.ddqrn.sess.run([self.ddqrn.predict, self.ddqrn.rnn_state], feed_dict={
                self.ddqrn.input_frames: [fv1],
                self.ddqrn.train_length: 1,
                self.ddqrn.batch_size: 1,
                self.ddqrn.state_in: self.rnn_state
            })
            a = a[0].item()

            self.a = a
            self.last_enemy_health = enemy_health
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
        sample = self.random_sample(self.number_of_offspring + 1)
        while sample:
            sample.sort(key=lambda x: x.fitness, reverse=True)
            self.population.append(sample[0])
            sample = self.random_sample(self.number_of_offspring + 1)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_individual = self.population[0]
        self.update_graph(best_individual)
        path = "./dqn"
        if not os.path.exists(path):
            os.makedirs(path)

        generation, summary = self.ddqrn.sess.run([self.ddqrn.inc_generation, self.fitness_summary],
                                                  feed_dict={self.fitness_tensor: best_individual.fitness})
        self.evaluation_rounds = cfg.eval_rounds(best_individual.fitness)
        self.base_population_size = cfg.population_size(best_individual.fitness)
        self.writer.add_summary(summary, generation)

        self.population.extend(self.generate_new_population())
        self.saver.save(self.ddqrn.sess, path + '/model-evolution')
        self.save_population()

    def random_sample(self, count) -> List[Individual]:
        res = []
        for k in range(count):
            if len(self.evaluated_population) > 0:
                i = random.randint(0, len(self.evaluated_population)-1)
                res.append(self.evaluated_population.pop(i))
        return res
