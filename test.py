# -*- coding: utf-8 -*-
import logging

import click

import settings
import task
import utils
import tensorflow as tf


_config = {}
_config['input_training_data_path'] = '../data'
_config['input_validation_data_path'] = '../data'
_config['input_previous_model_path'] = '../models'
_config['output_model_path'] =  '../models'
_config['log_dir'] = '../logs'
_config['node_count'] = 1
_config['node_list_path'] = ''
_config['node_id'] = 1
_config['node_name'] = ''
settings.public_config = _config
tf.gfile.MkDir(_config['output_model_path'])

config = {
'task' : 'Doc2VecProduct',
'news_encoder': 'cnnatt',
'arch' : 'avg',
'epochs' : 1,
'batch_size' : 50,
'training_step' : 2,
'validation_step' : 1,
'testing_impression' : 10,
'learning_rate' : 0.001,
'learning_rate_decay' : 0.2,
'gain' : 1.0,
'window_size' : 1,
'dropout' : 0.2,
'negative_samples' : 4,
'hidden_dim' : 200,
'nonlocal_negative_samples' : 0,
'enable_baseline' : True,
'title_filter_shape' : (400, 3),
'title_shape' : 5,
'body_shape' : 10,
'user_embedding_dim' : 200,
'personal_embedding_dim': 20,
'textual_embedding_dim' : 300,
'textual_embedding_trainable' : True,
'debug' : True,
'background' : False,
'name' : '',
'pretrain_name' : '',
'enable_pretrain_encoder' : False,
'pretrain_encoder_trainable' : True,

'body_sent_cnt': 10,
'body_sent_len': 20,
'body_filter_shape': (400,3),
'max_impression':200,
'max_impression_pos': 7,
'max_impression_neg': 200,
'test_window_size' : 1,
}

config = settings.Config(config)

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(config.log_output),
        logging.StreamHandler()
    ]
)

task_handler = task.get(config)

training_data = task_handler.train
#for i in range(100):
#    next(training_data)
#print(task_handler.training_step)
for epoch in range(config.epochs):
    logging.info('[+] start epoch {}'.format(epoch))
    model = task_handler.build_model(epoch)
    model.summary()
    history = model.fit_generator(
        training_data,
        1,
        # validation_data=task_handler.valid,
        # validation_steps=task_handler.validation_step,
        epochs=epoch + 1,
        initial_epoch=epoch,
        verbose=1 if config.debug and not config.background else 2)
    utils.logging_history(history)
    task_handler.callback(epoch)
    evaluations = model.evaluate_generator(
        task_handler.valid,
        task_handler.validation_step,
        verbose=1 if config.debug and not config.background else 2)
    utils.logging_evaluation(dict(zip(model.metrics_names, evaluations)))
    logging.info('[-] finish epoch {}'.format(epoch))

task_handler.save_model()
