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

sess = tf.Session()

ddqrn = DDQRN(sess, "main_DDQRN")
ddqrn_target = target_ddqrn(DDQRN(sess, "target_DDQRN"), tf.trainable_variables())

sess.run(tf.global_variables_initializer())

ddqrn_target.update(sess)  # Set the target network to be equal to the primary network

trainer = DDQRNTrainer(ddqrn, ddqrn_target, sess)

model = ModelSaver(ddqrn, trainer)

host = EvolutionHost("host", model)
population = [host.individual.generate_offspring(i) for i in range(cfg.population_size(0))]
ai_server = TournamentSelectionServer(ddqrn, population, model, trainer.train_writer)
#ai_server = AIServer(cfg.features, cfg.prediction_to_action, trainer, ddqrn, cfg.rew_funcs, model)

model.ai_server = ai_server



if cfg.load_model:
    model.load(cfg.save_path)

initial_count = ddqrn.sess.run([ddqrn.train_count])[0]
number_of_logs_to_train_on = 1000
print("Loading logs...")
logs = parse_logs_in_folder(cfg.log_folder)
print("Training on %s game logs" % len(logs))
start_time = time()
for p, log_file_pair in enumerate(logs):
    if p < initial_count:
        print("Skipping log number %s" % p)
        continue
    if p == number_of_logs_to_train_on:
        end_time = time()
        print("Training took %s seconds" % (end_time - start_time))
        exit(0)
    log_file_0, log_file_1 = log_file_pair
    print("Training on log number %s..." % p, end="", flush=True)
    trainer.start_episode()
    for i, event in enumerate(log_file_0):
        next_event = log_file_0[i + 1]
        # Observe play
        s = event.get_feature_vector()
        a = event.action
        s1 = next_event.get_feature_vector()
        r = next_event.reward

        end = next_event.end

        if r > 0:
            print(" Ooh reward!...", end="", flush=True)

        trainer.experience(s, a, r, s1, end)
        if end:
            break

    train_count = ddqrn.sess.run([ddqrn.inc_train_count])[0]
    #if train_count % 100 == 0:
    #    model.save(cfg.save_path)
    trainer.end_episode()
    print(" Done!")

    # Periodically save the model.
    if p % 50 == 0:
        model.save(cfg.save_path)
        print("Saved Model")

model.save(cfg.save_path)
print("Done training!")

# Assuming we have now done some kind of training.. Try to predict some actions!


server = SharpShooterServer()
server.start()
print("started Server")
i = ddqrn.sess.run([ddqrn.train_count])[0]
while True:
    server.receive_message(ai_server)
    if ai_server.game_has_ended:
        if i % 5000 == 0 and type(ai_server) is AIServer:
            ai_server.start_evaluation(100)
        i += 1
