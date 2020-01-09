import logging

import keras
import numpy as np
import tensorflow as tf

import document
import models
import settings
import utils


class UserEmbedding:
    queue_size = 100000

    def __init__(self, config: settings.Config):
        self.config = config
        self._load_docs()
        self._load_users()
        self._load_data()

    def _load_docs(self):
        logging.info("[+] loading docs metadata")
        parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.title_shape)
        )
        with utils.open(self.config.doc_meta_input) as file:
            docs = [line.split('\t') for line in file]

        self.docs = {int(line[1]): parser(line[4])[0] for line in docs}
        self.doc_count = max(self.docs.keys()) + 1
        self.doc_freq = np.zeros(self.doc_count)
        for line in docs:
            self.doc_freq[int(line[1])] = int(line[2])
        self.doc_freq = self.doc_freq ** 0.75
        self.doc_freq = self.doc_freq / np.sum(self.doc_freq)
        logging.info("[-] loaded {} docs".format(self.doc_count))

    def _load_users(self):
        logging.info("[+] loading users metadata")
        with utils.open(self.config.user_meta_input) as file:
            users = [line.split('\t') for line in file]
        self.users = {int(line[2]): line[0] for line in users}
        self.user_count = max(self.users.keys()) + 1
        logging.info("[-] loaded {} users".format(self.user_count))

    def _load_data(self):
        logging.info("[+] loading training data")
        with utils.open(self.config.training_data_input) as file:
            pairs = [line.split('\t') for line in file]
        pairs = np.array([[int(line[0]), int(line[1])] for line in pairs])
        self.partition = np.arange(len(pairs)) % 10 == 0
        self.training_pairs = pairs[~self.partition]
        self.validation_pairs = pairs[self.partition]
        self.training_pairs_count = len(self.training_pairs)
        self.validation_pairs_count = len(self.validation_pairs)
        self.training_step = np.ceil(self.training_pairs_count / self.config.batch_size)
        self.validation_step = np.ceil(self.validation_pairs_count / self.config.batch_size)
        logging.info("[-] loaded {}+{} pairs".format(self.training_pairs_count, self.validation_pairs_count))

    def _negative_sample(self):
        while True:
            yield from np.random.choice(self.doc_count, self.queue_size, p=self.doc_freq)

    @property
    def train(self):
        def __gen__():
            sampler = self._negative_sample()
            while True:
                np.random.shuffle(self.training_pairs)
                for u, e in self.training_pairs:
                    negs = [next(sampler) for _ in range(self.config.negative_samples)]
                    yield [u, self.docs[e]] + [self.docs[neg] for neg in negs]

        gen = __gen__()

        label = np.zeros((self.config.batch_size,))
        label = keras.utils.to_categorical(label, self.config.negative_samples + 1)

        while True:
            yield [np.stack(x) for x in zip(*(next(gen) for _ in range(self.config.batch_size)))], label

    @property
    def valid(self):
        def __gen__():
            sampler = self._negative_sample()
            while True:
                for u, e in self.validation_pairs:
                    negs = [next(sampler) for _ in range(self.config.negative_samples)]
                    yield [u, self.docs[e]] + [self.docs[neg] for neg in negs]

        gen = __gen__()

        label = np.zeros((self.config.batch_size,))
        label = keras.utils.to_categorical(label, self.config.negative_samples + 1)

        while True:
            yield [np.stack(x) for x in zip(*(next(gen) for _ in range(self.config.batch_size)))], label

    def build_model(self, epoch):
        if epoch == 0:
            self._build_model()
        return self.model

    def loss(self, y_true, y_pred):
        return -tf.reduce_mean(
            tf.reduce_sum(
                y_true * tf.log(y_pred + 1e-8) +
                (1 - y_true) * tf.log(1 - y_pred + 1e-8) / self.config.negative_samples,
                axis=-1))

    def acc(self, y_true, y_pred):
        return 0.5 * tf.reduce_mean(
            tf.reduce_sum(y_true * y_pred + (1 - y_true) * (1 - y_pred) / self.config.negative_samples, axis=-1)
        )

    def _build_model(self):
        title_embedding = utils.load_textual_embedding(self.config.title_embedding_input,
                                                       self.config.textual_embedding_dim)
        title_embedding_layer = keras.layers.Embedding(
            *title_embedding.shape,
            input_length=self.config.title_shape,
            weights=[title_embedding],
            trainable=self.config.textual_embedding_trainable
        )

        self.user_embedding_layer = user_embedding_layer = keras.layers.Embedding(
            self.user_count,
            self.config.user_embedding_dim,
            trainable=True
        )

        self.encoder = encoder = models.ca(
            self.config.title_shape,
            self.config.title_filter_shape,
            title_embedding_layer,
            self.config.dropout,
            self.config.user_embedding_dim
        )

        user = keras.Input((1,))
        title = [keras.Input((self.config.title_shape,)) for _ in range(self.config.negative_samples + 1)]

        user_vec = keras.layers.Reshape((-1,))(user_embedding_layer(user))
        title_vec = [encoder(t) for t in title]

        dist = [keras.layers.Dot((-1, -1))([keras.layers.Dropout(self.config.dropout)(user_vec),
                                            keras.layers.Dropout(self.config.dropout)(t_vec)]) for t_vec in title_vec]
        logits = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.concatenate(dist))

        self.model = keras.Model([user, title], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[self.acc]
        )

    def callback(self, epoch):
        pass

    def save_model(self):
        logging.info('[+] saving models')
        docs = {k: self.encoder.predict_on_batch(self.docs[k].reshape((1, -1)))[0] for k in self.docs}

        docs_matrix = np.array(
            [
                docs[i] if i in docs else
                np.zeros(self.config.user_embedding_dim, dtype=np.float32)
                for i in range(max(docs.keys()) + 1)
            ]
        )

        docs_embedding_layer = keras.layers.Embedding(
            *docs_matrix.shape,
            weights=[docs_matrix],
            trainable=False
        )
        user = keras.Input((1,))
        event = keras.Input((1,))
        dist = keras.layers.Reshape((1,))(keras.layers.Dot((-1, -1))([self.user_embedding_layer(user),
                                                                      docs_embedding_layer(event)]))
        logits = keras.layers.Activation(keras.activations.sigmoid)(dist)
        model = keras.Model([user, event], logits)

        utils.save_model(self.config.model_output, model)
        utils.save_model(self.config.encoder_output, self.encoder)
        logging.info('[-] saved models')
