import logging

import keras
import numpy as np
import tensorflow as tf

import document
import models
import settings
import utils


class TitleVertical:
    class News:
        __slots__ = ['vertical', 'sub_vertical', 'title', 'body']

        def __init__(self, vertical, sub_vertical, title, body):
            self.vertical = vertical
            self.sub_vertical = sub_vertical
            self.title = title
            self.body = body

    def __init__(self, config: settings.Config):
        self.config = config
        self._load_docs()
        self.training_step = len(self.train_index) // self.config.batch_size
        self.validation_step = len(self.valid_index) // self.config.batch_size

    def _load_docs(self):
        logging.info("[+] loading docs metadata")
        title_parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.title_shape)
        )
        body_parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.body_shape)
        )
        with open(self.config.doc_meta_input) as file:
            lines = [line.strip('\n').split('\t') for line in file]
        self.docs = [self.News(
            line[2],
            line[3],
            title_parser(line[4])[0],
            body_parser(line[5])[0],
        ) for line in lines]
        self.verticals = list(set(news.vertical for news in self.docs))
        self.sub_verticals = list(set(news.sub_vertical for news in self.docs))
        self.data_verticals = keras.utils.to_categorical(
            np.array([self.verticals.index(news.vertical) for news in self.docs]))
        self.data_titles = np.stack([news.title for news in self.docs])

        data = np.arange(len(self.docs))
        np.random.shuffle(data)
        self.train_index = data[:len(self.docs) // 10]
        self.valid_index = data[len(self.docs) // 10:]
        logging.info("[-] loaded docs metadata")

    @property
    def train(self):
        while True:
            np.random.shuffle(self.train_index)
            data_titles = self.data_titles[self.train_index]
            data_verticals = self.data_verticals[self.train_index]
            for batch in range(self.config.batch_size, len(self.train_index), self.config.batch_size):
                yield data_titles[batch - self.config.batch_size:batch], \
                      data_verticals[batch - self.config.batch_size:batch]

    @property
    def valid(self):
        while True:
            data_titles = self.data_titles[self.valid_index]
            data_verticals = self.data_verticals[self.valid_index]
            for batch in range(self.config.batch_size, len(self.valid_index), self.config.batch_size):
                yield data_titles[batch - self.config.batch_size:batch], \
                      data_verticals[batch - self.config.batch_size:batch]

    def build_model(self, epoch):
        if epoch == 0:
            self._build_model()
        return self.model

    def _build_model(self):
        if self.config.debug:
            title_embedding = np.load(self.config.title_embedding_input + '.npy')
        else:
            title_embedding = utils.load_textual_embedding(self.config.title_embedding_input,
                                                           self.config.textual_embedding_dim)
        title_embedding_layer = keras.layers.Embedding(
            *title_embedding.shape,
            input_length=self.config.title_shape,
            weights=[title_embedding],
            trainable=self.config.textual_embedding_trainable,
        )

        title = keras.Input((self.config.title_shape,))

        if self.config.arch == 'ca':
            self.title_encoder = title_encoder = models.ca(
                self.config.title_shape,
                self.config.title_filter_shape,
                title_embedding_layer,
                self.config.dropout)
            hidden = title_encoder(title)
        elif self.config.arch == 'la':
            self.title_encoder = title_encoder = models.la(
                self.config.title_shape,
                self.config.hidden_dim,
                title_embedding_layer,
                self.config.dropout)
            hidden = title_encoder(title)
        else:
            e = keras.layers.Dropout(self.config.dropout)(title_embedding_layer(title))
            c = keras.layers.Conv1D(*self.config.title_filter_shape, padding='same', activation='relu', strides=1)(e)
            hidden = keras.layers.GlobalAveragePooling1D()(keras.layers.Dropout(self.config.dropout)(c))

        logits = keras.layers.Dense(len(self.verticals), activation='softmax')(hidden)

        self.model = keras.Model(title, logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy])

    def callback(self, epoch):
        if epoch == 10 or epoch == 15:
            keras.backend.set_value(
                self.model.optimizer.lr,
                keras.backend.get_value(self.model.optimizer.lr) * self.config.learning_rate_decay)

    def save_model(self):
        if hasattr(self, 'title_encoder'):
            logging.info('[+] saving models')
            utils.save_model(self.config.encoder_output, self.title_encoder)
            logging.info('[-] saved models')
