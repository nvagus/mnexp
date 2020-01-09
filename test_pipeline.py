# -*- coding: utf-8 -*-

import tensorflow as tf
import settings
import task
import utils

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
'task' : 'TestPipelineBody',
'name': '',
'batch_size': 256,
'pipeline_input': '../../pipeline-2018-12-01'
}

config = settings.Config(config)

task_handler = task.get(config)
task_handler.load_model()
task_handler.test_doc_vec()
task_handler.test_user_vec()
task_handler.test_user_doc_score()
task_handler.test_correct()