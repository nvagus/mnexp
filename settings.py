import ast
import os
import pprint

import click
import tensorflow as tf

os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            if not isinstance(value, str):
                return value
            value = value.replace('/', '"')
            return ast.literal_eval(value)
        except Exception:
            raise click.BadParameter(value)


public_config = {}


class Config:
    __slots__ = [
        # input/output paths
        'input_training_data_path',
        'input_validation_data_path',
        'input_previous_model_path',
        'output_model_path',
        'log_dir',
        # global parameters
        'task',
        'evaluate_sessions',
        'enable_baseline',
        # training parameters
        'target',
        'epochs',
        'batch_size',
        'learning_rate',
        'learning_rate_decay',
        'training_step',
        'validation_step',
        'testing_impression',
        'validation_impression',
        # model structure
        'arch',
        'user_embedding_dim',
        'title_filter_shape',
        'title_shape',
        'body_shape',
        'window_size',
        'hidden_dim',
        'days',
        # model parameters
        'gain',
        'round',
        'dropout',
        'negative_samples',
        'nonlocal_negative_samples',
        'textual_embedding_dim',
        'textual_embedding_trainable',
        'enable_pretrain_encoder',
        'pretrain_encoder_trainable',
        # temporal parameters
        'name',
        'pretrain_name',
        'debug',
        'background',
        'personal_embedding_dim',
        'news_encoder',
        'use_vertical',
        'pipeline_input',
        'body_sent_cnt',
        'body_sent_len',
        'body_filter_shape',
        'max_impression',
        'max_impression_pos',
        'max_impression_neg',
        'score_model',
        'test_window_size',
        'vertical_embedding_dim',
        'subvertical_embedding_dim',
        'lrd_on_epochs',
        'id_keep',
        'use_generator',
        'user_feature_size',
        'doc_feature_size',
        'use_vertical_type',
    ]

    def __init__(self, config):
        config.update(public_config)
        for k, v in config.items():
            if not k.startswith('node'):
                setattr(self, k, v)
        pprint.pprint(config)
        try:
            print('Input Training Data Path:')
            pprint.pprint(tf.gfile.ListDirectory(self.input_training_data_path))
        except tf.errors.NotFoundError:
            pass

    @property
    def training_data_input(self):
        return os.path.join(self.input_training_data_path, 'ClickData.tsv')

    @property
    def testing_data_input(self):
        return os.path.join(self.input_training_data_path, 'TestData.tsv')

    @property
    def title_embedding_input(self):
        return os.path.join(self.input_training_data_path, 'Vocab.tsv')

    @property
    def doc_meta_input(self):
        return os.path.join(self.input_training_data_path, 'DocMeta.tsv')

    @property
    def user_meta_input(self):
        return os.path.join(self.input_training_data_path, 'UserMeta.tsv')

    @property
    def model_input(self):
        return os.path.join(
            self.input_previous_model_path, 'model{}.json'.format(self.pretrain_name)), os.path.join(
            self.input_previous_model_path, 'model{}.pkl'.format(self.pretrain_name))

    @property
    def model_output(self):
        return os.path.join(
            self.output_model_path, 'model{}.json'.format(self.name)), os.path.join(
            self.output_model_path, 'model{}.pkl'.format(self.name))

    @property
    def encoder_input(self):
        return os.path.join(
            self.input_previous_model_path, 'encoder{}.json'.format(self.pretrain_name)), os.path.join(
            self.input_previous_model_path, 'encoder{}.pkl'.format(self.pretrain_name))

    @property
    def encoder_output(self):
        return os.path.join(
            self.output_model_path, 'encoder{}.json'.format(self.name)), os.path.join(
            self.output_model_path, 'encoder{}.pkl'.format(self.name))

    @property
    def user_encoder_output(self):
        return os.path.join(
            self.output_model_path, 'user.encoder{}.json'.format(self.name)), os.path.join(
            self.output_model_path, 'user.encoder{}.pkl'.format(self.name))

    @property
    def result_output(self):
        return os.path.join(self.output_model_path, 'model{}.tsv'.format(self.name))

    @property
    def result_input(self):
        return os.path.join(self.input_previous_model_path, 'model{}.tsv'.format(self.pretrain_name))

    @property
    def log_output(self):
        return os.path.join(self.log_dir, 'log{}.txt'.format(self.name))

    @property
    def pipeline_inputs(self):
        docs_path = os.path.join(self.pipeline_input, 'docs.tsv')
        users_path = os.path.join(self.pipeline_input, 'UserClick.tsv')
        pair_path = os.path.join(self.pipeline_input, 'userDocPair.tsv')
        return docs_path, users_path, pair_path

    @property
    def pipeline_output(self):
        return os.path.join(self.pipeline_input, 'score_' + self.name + '.tsv')

    @property
    def doc_punc_index_input(self):
        return os.path.join(self.input_training_data_path, 'PuncIndex.tsv')

    @property
    def vertical2idx_input(self):
        return os.path.join(self.input_training_data_path, 'vertical2idx.tsv'), os.path.join(
            self.input_training_data_path, 'subvertical2idx.tsv')

    @property
    def train_npz_input(self):
        return os.path.join(self.input_training_data_path, 'train_{}days_{}window.npz').format(self.days,
                                                                                               self.window_size)

    @property
    def test_npz_input(self):
        return os.path.join(self.input_training_data_path, 'test_{}days_{}window.npz').format(self.days,
                                                                                              self.window_size)

    @property
    def train_sparse_input(self):
        return os.path.join(self.input_training_data_path, 'train_{}days_{}window_sparse.npz').format(self.days,
                                                                                               self.window_size)

    @property
    def test_sparse_input(self):
        return os.path.join(self.input_training_data_path, 'test_{}days_{}window_sparse.npz').format(self.days,
                                                                                              self.window_size)

    @property
    def vert_npz_input(self):
        return os.path.join(self.input_training_data_path, 'Vert.npz')


@click.group(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.option('--input-training-data-path', default='../data')
@click.option('--input-validation-data-path', default='../data')
@click.option('--input-previous-model-path', default='../models')
@click.option('--output-model-path', default='../models')
@click.option('--log-dir', default='../logs')
@click.option('--node-count', default=1)
@click.option('--node-list-path', default='')
@click.option('--node-id', default=1)
@click.option('--node-name', default='')
def main(**_config):
    global public_config
    public_config = _config
    tf.gfile.MkDir(_config['output_model_path'])
    return 0


def pass_config(f):
    def __wrapper__(**_config):
        _config = Config(_config)
        return f(_config)

    __wrapper__.__name__ = f.__name__

    return __wrapper__
