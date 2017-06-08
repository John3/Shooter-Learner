import json

import tensorflow as tf
from simple_ddqrn import DDQRN
from target_ddqrn import target_ddqrn
from ddqrn_trainer import DDQRNTrainer
from model_saver import ModelSaver

import parameter_config as cfg

class GrandTournamentServer:

    def __init__(self, matches_per_round=100):

        self.ai_participants = [
            {"name": "FreksenThink", "rlAi": False},
            {"name": "FriskPige", "rlAi": False},
            {"name": "ScottSteiner", "rlAi": False},
            {"name": "Turing", "rlAi": False},
            {"name": "BAI", "rlAi": False},
            #{"name": "ScannerBot", "rlAi": False},
            #{"name": "Kurt", "rlAi": False},
            #{"name": "HundenBider", "rlAi": False},
            {"name": "RLFreksenThink", "rlAi": True},
            {"name": "RLBAI", "rlAi": True},
            {"name": "RLScottSteiner", "rlAi": True},
            {"name": "RLMix", "rlAi": True},
            {"name": "RLCopy", "rlAi": True},
            {"name": "RLBAIEvo", "rlAi": True},
            {"name": "RLPunisher", "rlAi": True},
        ]

        rl_ais = [x for x in self.ai_participants if x["rlAi"] is True]
        for ai in rl_ais:
            self.load_and_add_rl_participant(ai["name"])


        self.player_one_set = False
        self.player_two_set = False
        self.match_finished = True
        self.match_count = 0
        self.matches_per_round = matches_per_round
        self.current_player = 0
        self.current_opponent = 0
        self.score = [0, 0]

    def callback(self, msg):
        if msg["type"] == "instruction":
            if not self.player_two_set:
                self.current_opponent += 1
                if not self.current_opponent < len(self.ai_participants):
                    self.current_player += 1
                    self.current_opponent = self.current_player + 1

                    if not self.current_opponent < len(self.ai_participants):
                        exit()
                opponent = self.ai_participants[self.current_opponent]
                self.player_two_set = True
                print("Setting AI 1 to %s " % opponent["name"])
                return {
                    "type": "instruction",
                    "command": "setAi",
                    "playerNumber": 1,
                    "thinkFunction": "RL_AIClient_Think1" if opponent["rlAi"] else opponent["name"]
                }

            if not self.player_one_set:
                player = self.ai_participants[self.current_player]
                self.player_one_set = True
                print("Setting AI 0 to %s " % player["name"])
                return {
                    "type": "instruction",
                    "command": "setAi",
                    "playerNumber": 0,
                    "thinkFunction": "RL_AIClient_Think0" if player["rlAi"] else player["name"]
                }

            return {"type": "instruction", "command": "resetMission"}

        if msg["type"] == "event":
            if msg["message"] == "game_start":
                return {"response": "success"}

            if msg["message"] == "game_end":
                if msg["result"] != "draw":
                    winner = int(msg["result"][6:7])
                    self.score[winner] += 1
                else:
                    winner = "no-one"
                print("\r%s VS %s: %d -- %d [%d]  " % (
                    self.ai_participants[self.current_player]["name"],
                    self.ai_participants[self.current_opponent]["name"],
                    self.score[0],
                    self.score[1],
                    self.match_count
                ), end="")
                self.match_count += 1
                if self.match_count == self.matches_per_round:
                    print()
                    print("Round finished, score was %s" % self.score)
                    self.match_count = 0
                    self.score = [0, 0]
                    self.player_one_set = False
                    self.player_two_set = False

                return {"response": "success"}

        if msg["type"] == "think":
            player = int(msg["playerNumber"])
            fv = self.json_string_to_feature_vector(msg["feature_vector"])

            ai = self.ai_participants[player]

            a, ai["ddqrn"].state = ai["ddqrn"].get_prediction_with_state(
                input=[fv],
                train_length=1,
                state_in=ai["ddqrn"].state,
                batch_size=1
            )
            a = a[0].item()
            return cfg.prediction_to_action[a]

        print("Unhandled message: " + str(msg))
        return 3

    def json_string_to_feature_vector(self, json_str):
        fv = json.loads(json_str)
        out = []
        for feature in cfg.features:
            out.append(fv[feature])
        return out

    def load_and_add_rl_participant(self, name):
        sess = tf.Session()

        ddqrn = DDQRN(sess, "main_DDQRN")
        ddqrn_target = target_ddqrn(DDQRN(sess, "target_DDQRN"),
                                    [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_DDQRN"),
                                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_DDQRN")])

        sess.run(tf.global_variables_initializer())

        trainer = DDQRNTrainer(ddqrn, ddqrn_target, sess)

        model = ModelSaver(ddqrn, trainer)

        model.load("torunament_models/%s" % name)

        next(x for x in self.ai_participants if x["name"] == name)["sess"] = sess
        next(x for x in self.ai_participants if x["name"] == name)["ddqrn"] = ddqrn

        return sess, ddqrn
