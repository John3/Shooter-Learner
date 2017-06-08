import tensorflow as tf
from time import time

from ai_server import AIServer
from ddqrn_trainer import DDQRNTrainer
from evolution_trainer import EvolutionHost
from log_parser import parse_logs_in_folder
from model_saver import ModelSaver
from sharpshooter_server import SharpShooterServer
from simple_ddqrn import DDQRN
from target_ddqrn import target_ddqrn
import parameter_config as cfg
from tournament_selection_server import TournamentSelectionServer
import numpy as np
import os

sess = tf.Session()

ddqrn = DDQRN(sess, "main_DDQRN")
ddqrn_target = target_ddqrn(DDQRN(sess, "target_DDQRN"),
                            [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_DDQRN"),
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_DDQRN")])

sess.run(tf.global_variables_initializer())

#ddqrn_target.update(sess)  # Apparently not necessary

trainer = DDQRNTrainer(ddqrn, ddqrn_target, sess)

model = ModelSaver(ddqrn, trainer)

def logtraining(logs, time_file, action_percentage_file):
    for p, log_file_pair in enumerate(logs):
        start_time = time()
        ##if p < initial_count:
        ##    print("Skipping log number %s" % p)
        ##    continue
            
        log_file_0, log_file_1 = log_file_pair
        print("Training on log number %s..." % p, end="", flush=True)
        trainer.start_episode()
        last_action = None
        last_enemy_health = None
        for i, event in enumerate(log_file_0):
            next_event = log_file_0[i + 1]
            # Observe play
            s = event.get_feature_vector()
            a = event.action
            s1 = next_event.get_feature_vector()
            r = next_event.reward
            
            end = next_event.end
            # predict network action
            a_net, ddqrn.state = ddqrn.get_prediction_with_state(
                    input=[s],
                    train_length=1,
                    state_in=ddqrn.state,
                    batch_size=1
                )
            a_net = a_net[0].item()
            #'''
            if a == a_net:
                r = 1
            else:
                r = -1
            #'''
            '''
            if (action  == "shoot" and s[14] > 0) or
            (action  == "none") or
            (prepared && action == "prepare"):
                if prepared:
                    prepared = False
                network_reward = -1
                
            if cfg.action_to_string(last_action) == "prepare":
                prepared = True
            #experience predicted network action
            '''
            '''
            if r == 0:
                r = cfg.meta_reward(last_action,last_enemy_health,s)
            elif r == 1:
                r = 100000
            if r > 0:
                print(" Ooh reward!...", end="", flush=True)
            '''    
            trainer.experience(s, a_net, r, s1, end)
            last_action = a
            last_enemy_health = s[10]
            if end:
                break

        train_count = ddqrn.sess.run([ddqrn.inc_train_count])[0]
        trainer.end_episode()
        print(" Done!")
        end_time = time()
        print("Training took %s seconds" % (end_time - start_time))
        time_file.write("%s %s\n" % (p, end_time - start_time))
        
        # Periodically save the model.
        #Evaluate the log model
        if p % 50 == 0:
            model.save(cfg.save_path)
            print("Saved Model")
            log_eval(logs,100,action_percentage_file)
            
def log_eval(logs,num_eval,action_percentage_file):
    Ai_actions = np.array([])
    predicted_actions = np.array([])
    num_actions = 0
    same_actions = 0
    for p, log_file_pair in enumerate(logs):
        if p >= num_eval:
            break
        log_file_0, log_file_1 = log_file_pair
        print("evaluating log number %s..." % p, end="", flush=True)
        for i, event in enumerate(log_file_0):
            s = event.get_feature_vector()
            a = event.action
            a_net, ddqrn.state = ddqrn.get_prediction_with_state(
                    input=[s],
                    train_length=1,
                    state_in=ddqrn.state,
                    batch_size=1
                )
            a_net = a_net[0].item()
            num_actions = num_actions + 1
            if a_net == a:
                same_actions = same_actions + 1
            Ai_actions = np.append(Ai_actions, a)
            predicted_actions = np.append(predicted_actions, a_net)

    action_percentage_file.write("%s\n"% (same_actions/num_actions))
    i = 0
    path_hist = "data/Histogram_" + cfg.run_name + "_"        
    while os.path.isfile(path_hist + str(i) + '.dat'):
        i = i+1
    with open(path_hist + str(i) + '.dat', 'w') as histogram_file:
        histogram_file.write("#Action AI_action Predicted_Action\n")
        for i in range(9):
            p = len(np.extract(Ai_actions==i,Ai_actions))
            q = len(np.extract(predicted_actions == i,predicted_actions))
            histogram_file.write("%s %s %s\n"% (cfg.action_to_string[i], p, q))
    

            
if cfg.load_model:
    model.load(cfg.save_path)

###Initial Log training###    
initial_count = ddqrn.sess.run([ddqrn.train_count])[0]
print("Loading logs...")
logs = parse_logs_in_folder(cfg.log_folder)
print("Training on %s game logs" % len(logs))
time_file = open('times.dat', 'w')
#logtraining(logs,time_file)
time_file.close()
model.save(cfg.save_path)
print("Done training!")


if cfg.server == cfg.evolution:
    host = EvolutionHost("host", model)
    population = [host.individual.generate_offspring(i) for i in range(cfg.population_size(0))]
    ai_server = TournamentSelectionServer(ddqrn, ddqrn_target, population, model, trainer.train_writer)
elif cfg.server == cfg.gradient:
    ai_server = AIServer(cfg.features, cfg.prediction_to_action, trainer, ddqrn, cfg.rew_funcs, model)
elif cfg.server == cfg.logtraining:
    ###Continuing Log Training###
    with open('data/Logs_'+cfg.run_name+'.dat','w') as time_f:
        with open('data/Action_persentage_'+cfg.run_name+'.dat', 'w') as persentage_file:
            while True:
                logtraining(logs,time_f,persentage_file)
        
    
model.ai_server = ai_server


# Assuming we have now done some kind of training.. Try to predict some actions!


server = SharpShooterServer()
server.start()
print("started Server")
i = ddqrn.sess.run([ddqrn.train_count])[0]
while True:
    server.receive_message(ai_server)
    if ai_server.game_has_ended:
        if i % 500 == 0 and type(ai_server) is AIServer:
            ai_server.start_evaluation(200)
        i += 1
