import tensorflow as tf
from simple_ddqrn import DDQRN
from target_ddqrn import target_ddqrn

import parameter_config as cfg
from model_saver import ModelSaver
from ddqrn_trainer import DDQRNTrainer

sess = tf.Session()

ddqrn = DDQRN(sess, "main_DDQRN")
ddqrn_target = target_ddqrn(DDQRN(sess, "target_DDQRN"),
                            [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="main_DDQRN"),
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_DDQRN")])

sess.run(tf.global_variables_initializer())

trainer = DDQRNTrainer(ddqrn, ddqrn_target, sess)

model = ModelSaver(ddqrn, trainer)

model.load(cfg.save_path)
ddqrn_target.update(sess, tau=1.0)
model.save(cfg.save_path)
