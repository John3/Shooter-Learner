import tensorflow as tf

from ai_server import AIServer
from ddqrn_trainer import DDQRNTrainer
from log_parser import parse_logs_in_folder
from sharpshooter_server import SharpShooterServer
from simple_ddqrn import DDQRN
from target_ddqrn import target_ddqrn
import parameter_config as cfg

sess = tf.Session()

ddqrn = DDQRN(sess, len(cfg.features), len(cfg.prediction_to_action), "main_DDQRN")
ddqrn_target = target_ddqrn(
    DDQRN(sess, len(cfg.features), len(cfg.prediction_to_action), "target_DDQRN"),
    tf.trainable_variables()
)

sess.run(tf.global_variables_initializer())

ddqrn_target.update(sess)  # Set the target network to be equal to the primary network

load_model = False
save_path = "./dqn"

trainer = DDQRNTrainer(ddqrn, ddqrn_target, sess, cfg.batch_size, cfg.trace_length)

player_number = 0

if load_model:
    trainer.load(save_path)
else:
    print("Loading logs...")
    logs = parse_logs_in_folder("data/game_logs")
    print("Training on %s game logs" % len(logs))
    for p, log_file_pair in enumerate(logs):
        log_file_0, log_file_1 = log_file_pair
        print("Training on log number %s..." % p, end="", flush=True)
        trainer.start_episode()
        for i, event in enumerate(log_file_0):
            next_event = log_file_0[i + 1]
            # Observe play
            s = event.get_feature_vector(cfg.features)
            a = event.action
            s1 = next_event.get_feature_vector(cfg.features)
            r = next_event.reward

            end = next_event.end

            if r > 0:
                print(" Ooh reward!...", end="", flush=True)

            trainer.experience(s, a, r, s1, end)
            if end:
                break

        train_count = ddqrn.sess.run([ddqrn.inc_train_count])[0]
        if train_count % 10 == 0:
            trainer.save("./dqn")
        trainer.end_episode()
        print(" Done!")

        # Periodically save the model.
        if p % 5 == 0:
            trainer.save(save_path)
            print("Saved Model")

trainer.save(save_path)
print("Done training!")


# Assuming we have now done some kind of training.. Try to predict some actions!

ai_server = AIServer(cfg.features, cfg.prediction_to_action, trainer, ddqrn)

server = SharpShooterServer()
server.start()
i = 1
while True:
    server.receive_message(ai_server)
    if ai_server.game_has_ended:
        if i % 50000 == 0:
            ai_server.start_evaluation(1000)
        i += 1
