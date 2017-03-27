import tensorflow as tf

from ai_server import AIServer
from ddqrn_trainer import DDQRNTrainer
from log_parser import parse_logs_in_folder
from sharpshooter_server import SharpShooterServer
from simple_ddqrn import DDQRN
from target_ddqrn import target_ddqrn
import parameter_config as cfg

action_to_string = {
    0: "none",
    1: "moveForward",
    2: "moveLeft",
    3: "moveRight",
    4: "moveBackward",
    5: "turnLeft",
    6: "turnRight",
    7: "shoot",
    8: "prepare"
}

prediction_to_action = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sess = tf.Session()

ddqrn = DDQRN(sess, cfg.fv_size, cfg.fv_size, cfg.actions_size, "main_DDQRN")
ddqrn_target = target_ddqrn(DDQRN(sess, cfg.fv_size, cfg.fv_size, cfg.actions_size, "target_DDQRN"), tf.trainable_variables())

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
            s = event.feature_vector
            a = event.action
            s1 = next_event.feature_vector
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


ai_server = AIServer(cfg.fv_size, cfg.actions_size, trainer, ddqrn)

server = SharpShooterServer()
server.start()
i = 1
while True:
    server.receive_message(ai_server)
    if ai_server.game_has_ended:
        if i % 50000 == 0:
            ai_server.start_evaluation(1000)
        i += 1
