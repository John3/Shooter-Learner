import json

import numpy as np
import tensorflow as tf


class AIServer:

    def __init__(self, fv_size, prediction_to_action, trainer, ddqrn):
        self.fv_size = fv_size
        self.prediction_to_action = prediction_to_action
        self.trainer = trainer
        self.ddqrn = ddqrn

        self.training = True
        self.fv0 = None
        self.a = None
        self.eval_games = 0
        self.num_games = 0
        self.rewards = 0
        self.result_tensor = tf.placeholder(tf.float32)
        self.result_summary = tf.summary.scalar('estimated_value', self.result_tensor)
        self.action_tensor = tf.placeholder(tf.int32)
        self.action_summary = tf.summary.histogram('selected_action', self.action_tensor)
        self.last_enemy_health = 20

        self.game_has_ended = False

    def start_evaluation(self, eval_games):
        self.training = False
        self.num_games = 0
        self.rewards = 0
        self.eval_games = eval_games

    def end_evaluation(self):
        i = self.ddqrn.sess.run([self.ddqrn.evaluation_count])[0]
        self.training = True
        value = self.rewards / self.num_games
        _, summary = self.ddqrn.sess.run([self.result_tensor, self.result_summary], feed_dict={self.result_tensor: value})
        self.trainer.test_writer.add_summary(summary, i)

    def callback(self, msg):
        self.game_has_ended = False
        if msg["type"] == "instruction":
            return {"type": "instruction", "command": "resetMission"}

        if msg["type"] == "event":
            if msg["message"] == "game_start":
                # Reset state
                self.ddqrn.state = (np.zeros([1, self.fv_size]), np.zeros([1, self.fv_size]))
                if self.training:
                    self.trainer.start_episode()

                print("Game has started!")
                print("%s" % (self.ddqrn.sess.run([self.ddqrn.train_count])))
                return {"response": "success"}
            if msg["message"] == "game_end":
                self.game_has_ended = True
                print("Game ended with result: " + str(msg))

                winner = msg["result"]
                r = 0
                if winner.startswith("player0"):
                    r = 1

                if self.training:
                    train_count = self.ddqrn.sess.run([self.ddqrn.inc_train_count])[0]
                    self.trainer.experience(self.fv0, self.a, r, self.fv0, True)
                    self.trainer.end_episode()
                    if train_count % 10 == 0:
                        self.trainer.save("./dqn")
                else:
                    self.ddqrn.sess.run([self.ddqrn.inc_evaluation_count])
                    self.num_games += 1
                    self.rewards += r
                    if self.num_games == self.eval_games:
                        self.end_evaluation()

                return {"response": "success"}

        if msg["type"] == "think":
            fv1 = self.json_string_to_feature_vector(msg["feature_vector"])

            enemy_health = fv1[10]

            r = 0
            if self.a == 7 and enemy_health < self.last_enemy_health:
                r = 0.00001

            if self.fv0 is not None and self.training:
                self.trainer.experience(self.fv0, self.a, r, fv1, False)

            if np.random.rand(1) < self.trainer.e or self.trainer.total_steps < self.trainer.pre_train_steps:
                self.ddqrn.state = self.ddqrn.get_state(
                    input=[fv1],
                    train_length=1,
                    state_in=self.ddqrn.state,
                    batch_size=1
                )
                a = np.random.randint(0, len(self.prediction_to_action))
            else:
                a, self.ddqrn.state = self.ddqrn.get_prediction_with_state(
                    input=[fv1],
                    train_length=1,
                    state_in=self.ddqrn.state,
                    batch_size=1
                )
                a = a[0].item()

            if not self.training:
                i = self.ddqrn.sess.run([self.ddqrn.evaluation_count])[0]
                _, summary = self.ddqrn.sess.run([self.action_tensor, self.action_summary],
                                                 feed_dict={self.action_tensor: a})
                self.trainer.test_writer.add_summary(summary, i)

            self.fv0 = fv1
            self.a = a
            self.last_enemy_health = enemy_health
            return self.prediction_to_action[a]

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
