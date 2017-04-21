import os

import tensorflow as tf
import numpy as np
import parameter_config as cfg
from tournament_selection_server import TournamentSelectionServer


class ModelSaver:

    def __init__(self, ddqrn, trainer):
        self.ddqrn = ddqrn
        self.trainer = trainer
        self.ai_server = None

        self.saver = tf.train.Saver()

    def load(self, path):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.ddqrn.sess, ckpt.model_checkpoint_path)
        self.trainer.buffer.load(path)
        #self.trainer.total_steps = cfg.pre_train_steps

        if type(self.ai_server) is TournamentSelectionServer:
            if os.path.exists(path + "/evolution.npz"):
                with np.load(path + "/evolution.npz") as data:
                    self.ai_server.population = list(data['population'])
                    self.ai_server.evaluated_population = list(data['evaluated_population'])
                self.ai_server.current_individual = self.ai_server.population.pop()

    def save(self, path):
        # Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)

        self.saver.save(self.ddqrn.sess, path + '/model')
        self.trainer.buffer.save(path)

        if type(self.ai_server) is TournamentSelectionServer:
            np.savez_compressed(path + "/evolution", population=self.ai_server.population,
                                evaluated_population=self.ai_server.evaluated_population)

    def save_population(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(path + "/evolution", population=self.ai_server.population,
                            evaluated_population=self.ai_server.evaluated_population)

    def restore(self, sess, path):
        self.saver.restore(sess, path)
