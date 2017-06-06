import json

import sys

import parameter_config as cfg

class GrandTournamentServer:

    def __init__(self, matches_per_round=100):
        self.learner_ai = {
            "RLFreksenThink": "asd",
            "RLTuring": "asd",
            "RLBAI": "asd",
            "RLMix": "asd",
            "RLCopy": "asd"
            # etc..
        }

        self.ai_participants = [
            {"name": "FreksenThink", "rlAi": False},
            {"name": "FriskPige", "rlAi": False},
            {"name": "ScottSteiner", "rlAi": False},
            {"name": "Turing", "rlAi": False},
            {"name": "BAI", "rlAi": False},
            {"name": "ScannerBot", "rlAi": False},
            {"name": "Kurt", "rlAi": False},
            {"name": "HundenBider", "rlAi": False},
            #{"name": "RLFreksenThink", "rlAi": True},
            #{"name": "RLTuring", "rlAi": True},
            #{"name": "RLBAI", "rlAi": True},
            #{"name": "RLMix", "rlAi": True},
            #{"name": "RLCopy", "rlAi": True}
        ]

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
                return {
                    "type": "instruction",
                    "command": "setAi",
                    "playerNumber": 1,
                    "thinkFunction": "RL_AIClient_Think1" if opponent["rlAi"] else opponent["name"]
                }

            if not self.player_one_set:
                player = self.ai_participants[self.current_player]
                self.player_one_set = True
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
                    self.ai_participants[self.current_player],
                    self.ai_participants[self.current_opponent],
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
            player = msg["playerNumber"]
            fv = self.json_string_to_feature_vector(msg["feature_vector"])

        print("Unhandled message: " + str(msg))
        return 3

    def json_string_to_feature_vector(self, json_str):
        fv = json.loads(json_str)
        out = []
        for feature in cfg.features:
            out.append(fv[feature])
        return out
