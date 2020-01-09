# -*- coding: utf-8 -*-
from .seq2vec import *
import logging

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import document
import models
import settings
import utils


class Seq2VecEncoder(Seq2VecLongEncoder):
    def get_doc_encoder(self):
        return self._get_doc_encoder(self.config.title_shape)

    def _get_doc_encoder(self, input_shape=None):
        if self.config.enable_pretrain_encoder:
            encoder = utils.load_model(self.config.encoder_input)
            if not self.config.pretrain_encoder_trainable:
                encoder.trainable = False
            return encoder
        else:
            if self.config.debug:
                title_embedding = np.load(self.config.title_embedding_input + '.npy')
            else:
                title_embedding = utils.load_textual_embedding(self.config.title_embedding_input,
                                                               self.config.textual_embedding_dim)

            news_model = self.config.news_encoder

            inp_shape = input_shape

            if news_model == 'cnnatt':
                mask_zero = False
            elif news_model == 'gruatt':
                mask_zero = True
            elif news_model == 'lstmatt':
                mask_zero = True
            elif news_model == 'bigruatt':
                mask_zero = True
            elif news_model == 'bilstmatt':
                mask_zero = True
            else:
                raise Exception('Unsupport doc model')

            title_embedding_layer = keras.layers.Embedding(
                *title_embedding.shape,
                input_length=inp_shape,
                weights=[title_embedding],
                trainable=self.config.textual_embedding_trainable,
                mask_zero=mask_zero
            )

            if news_model == 'cnnatt':

                inp = keras.Input((inp_shape,), dtype='int32')
                emb = title_embedding_layer(inp)

                filter_count, filter_size = self.config.title_filter_shape

                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1,
                                          use_bias=False)

                emb = keras.layers.Dropout(self.config.dropout)(emb)
                c = cnn(emb)

                cmask = keras.layers.Masking()(c)
                a = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask))
                #                a = models.SimpleAttentionMaskSupport()(cmask)
                #                a = models.simple_attention(keras.layers.Dropout(self.config.dropout)(c))

                a = keras.layers.Dense(self.config.user_embedding_dim)(a)

                doc_encoder = keras.Model(inp, a, name='doc_encoder')


            elif news_model == 'gruatt':
                inp = keras.Input((inp_shape,), dtype='int32')
                emb = title_embedding_layer(inp)
                rnnout = keras.layers.GRU(self.config.user_embedding_dim, return_sequences=True)(emb)
                doc_vec = models.Attention()(rnnout)
                doc_encoder = keras.Model(inp, doc_vec, name='doc_encoder')
            elif news_model == 'lstmatt':
                inp = keras.Input((inp_shape,), dtype='int32')
                emb = title_embedding_layer(inp)
                rnnout = keras.layers.LSTM(self.config.user_embedding_dim, return_sequences=True)(emb)
                doc_vec = models.Attention()(rnnout)
                doc_encoder = keras.Model(inp, doc_vec, name='doc_encoder')
            elif news_model == 'bigruatt':
                if self.config.user_embedding_dim % 2 != 0:
                    raise Exception('Unsupport bigruatt model. user_embedding_dim%2 must be zero')
                inp = keras.Input((inp_shape,), dtype='int32')
                emb = title_embedding_layer(inp)
                rnnout = keras.layers.Bidirectional(
                    keras.layers.GRU(self.config.user_embedding_dim // 2, return_sequences=True), merge_mode='concat')(
                    emb)
                doc_vec = models.Attention()(rnnout)
                doc_encoder = keras.Model(inp, doc_vec, name='doc_encoder')
            elif news_model == 'bilstmatt':
                if self.config.user_embedding_dim % 2 != 0:
                    raise Exception('Unsupport bilstmatt model. user_embedding_dim%2 must be zero')
                inp = keras.Input((inp_shape,), dtype='int32')
                emb = title_embedding_layer(inp)
                rnnout = keras.layers.Bidirectional(
                    keras.layers.LSTM(self.config.user_embedding_dim // 2, return_sequences=True), merge_mode='concat')(
                    emb)
                doc_vec = models.Attention()(rnnout)
                doc_encoder = keras.Model(inp, doc_vec, name='doc_encoder')

            else:
                raise Exception('Unsupport doc model')

            return doc_encoder

    def get_user_encoder(self, window_size=None):
        if window_size is None:
            window_size = self.config.window_size

        user_clicked_vec = keras.Input((window_size, self.config.user_embedding_dim), name='user_clicked_vec')

        user_model = self.config.arch

        if user_model == 'att':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = models.SimpleAttentionMaskSupport()(mask_clicked_vec)
        elif user_model == 'gruatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            mask_clicked_vec = keras.layers.GRU(self.config.user_embedding_dim, return_sequences=True)(mask_clicked_vec)
            user_vec = models.Attention()(mask_clicked_vec)
        elif user_model == 'avg':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = models.GlobalAveragePoolingMaskSupport()(mask_clicked_vec)
        else:
            raise Exception('Unsupport user model')

        user_encoder = keras.Model(user_clicked_vec, user_vec, name='user_encoder')

        return user_encoder

    def score_encoder(self, user_vec, candidate_vec):
        join_vec = keras.layers.concatenate([user_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.hidden_dim, activation='relu', name='concat_dense')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid', name='socre_dense')(hidden)
        return logits

    def modeling(self, input_shape):
        inp_shape = input_shape

        clicked = keras.Input((self.config.window_size, inp_shape))
        doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])

        user_encoder = self.get_user_encoder()

        user_vec = user_encoder(clicked_vec)

        candidate = keras.Input((inp_shape,))
        candidate_vec = doc_encoder(candidate)

        logits = self.score_encoder(user_vec, candidate_vec)

        self.model = keras.Model([clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )

    def _build_model(self):
        self.modeling(self.config.title_shape)

    def valid_gen(self):
        return self._valid_gen('title')

    def _valid_gen(self, title_or_body='title'):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2] and line[3]:
                        ih1 = self._extract_impressions(line[2])
                        ih2 = self._extract_impressions(line[3])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih1:
                            for pos in impression.pos:
                                ch.push(pos)
                        for impression in ih2:
                            for pos in impression.pos:
                                if ch.count:
                                    if title_or_body == 'title':

                                        clicked = ch.get_title()
                                        yield clicked, self.docs[pos].title, 1
                                        for neg in impression.negative_samples(self.config.negative_samples):
                                            yield clicked, self.docs[neg].title, 0
                                    elif title_or_body == 'body':
                                        clicked = ch.get_body()
                                        yield clicked, self.docs[pos].body, 1
                                        for neg in impression.negative_samples(self.config.negative_samples):
                                            yield clicked, self.docs[neg].body, 0
                                    elif title_or_body == 'doc':
                                        clicked = ch.get_doc()
                                        yield clicked, self.docs[pos].doc, 1
                                        for neg in impression.negative_samples(self.config.negative_samples):
                                            yield clicked, self.docs[neg].doc, 0
                                    elif title_or_body == 'concat':
                                        clicked_title = ch.get_title()
                                        clicked_body = ch.get_body()
                                        clicked = np.concatenate([clicked_title, clicked_body], axis=-1)
                                        candidate = np.concatenate([self.docs[pos].title, self.docs[pos].body], axis=-1)
                                        yield clicked, candidate, 1
                                        for neg in impression.negative_samples(self.config.negative_samples):
                                            candidate = np.concatenate([self.docs[neg].title, self.docs[neg].body],
                                                                       axis=-1)
                                            yield clicked, candidate, 0

                                    elif title_or_body == 'title_body':
                                        clicked_title = ch.get_title()
                                        clicked_body = ch.get_body()
                                        yield clicked_title, clicked_body, self.docs[pos].title, self.docs[pos].body, 1
                                        for neg in impression.negative_samples(self.config.negative_samples):
                                            yield clicked_title, clicked_body, self.docs[neg].title, self.docs[
                                                neg].body, 0
                                ch.push(pos)

    def test_gen(self):
        return self._test_gen('title')

    def _test_gen(self, title_or_body='title'):
        def _gen_title(clicked, impression):
            for p in impression.pos:
                doc = self.docs[p]
                yield clicked, doc.title, 1
            for n in impression.neg:
                doc = self.docs[n]
                yield clicked, doc.title, 0

        def _gen_body(clicked, impression):
            for p in impression.pos:
                doc = self.docs[p]
                yield clicked, doc.body, 1
            for n in impression.neg:
                doc = self.docs[n]
                yield clicked, doc.body, 0

        def _gen_doc(clicked, impression):
            for p in impression.pos:
                doc = self.docs[p]
                yield clicked, doc.doc, 1
            for n in impression.neg:
                doc = self.docs[n]
                yield clicked, doc.doc, 0

        def _gen_concat(clicked, impression):
            for p in impression.pos:
                candidate = np.concatenate([self.docs[p].title, self.docs[p].body], axis=-1)
                yield clicked, candidate, 1
            for n in impression.neg:
                candidate = np.concatenate([self.docs[n].title, self.docs[n].body], axis=-1)
                yield clicked, candidate, 0

        def _gen_title_body(clicked_title, clicked_body, impression):
            for p in impression.pos:
                doc = self.docs[p]
                yield clicked_title, clicked_body, doc.title, doc.body, 1
            for n in impression.neg:
                doc = self.docs[n]
                yield clicked_title, clicked_body, doc.title, doc.body, 0

        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                if line[2] and line[3]:
                    ih1 = self._extract_impressions(line[2])
                    ih2 = self._extract_impressions(line[3])
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        if title_or_body == 'title':
                            clicked = ch.get_title()
                            yield list(_gen_title(clicked, impression))

                        elif title_or_body == 'body':
                            clicked = ch.get_body()
                            yield list(_gen_body(clicked, impression))
                        elif title_or_body == 'doc':
                            clicked = ch.get_doc()
                            yield list(_gen_doc(clicked, impression))

                        elif title_or_body == 'concat':
                            clicked_title = ch.get_title()
                            clicked_body = ch.get_body()
                            clicked = np.concatenate([clicked_title, clicked_body], axis=-1)
                            yield list(_gen_concat(clicked, impression))

                        elif title_or_body == 'title_body':
                            clicked_title = ch.get_title()
                            clicked_body = ch.get_body()
                            yield list(_gen_title_body(clicked_title, clicked_body, impression))

                        for pos in impression.pos:
                            ch.push(pos)


class Seq2VeRandom(Seq2VecEncoder):
    class DocsWindow:
        def __init__(self, docs, window_size):

            self.docs = docs
            self.window_size = window_size

        def get_title(self, seq):
            if len(seq) < self.window_size:
                seq = np.pad(seq, (self.window_size - len(seq), 0), mode='constant')

            return np.stack([self.docs[i].title for i in seq])

        def get_body(self, seq):
            if len(seq) < self.window_size:
                seq = np.pad(seq, (self.window_size - len(seq), 0), mode='constant')

            return np.stack([self.docs[i].body for i in seq])

        def get_doc(self, seq):
            if len(seq) < self.window_size:
                seq = np.pad(seq, (self.window_size - len(seq), 0), mode='constant')

            return np.stack([self.docs[i].doc for i in seq])

    def _load_data(self):
        self._training_step = None
        self.validation_step = self.config.validation_step

    @property
    def training_step(self):
        if self._training_step:
            return self._training_step
        cnt = 0
        gen = self.train_gen(once=True)
        for data in gen:
            cnt = cnt + 1

        step = cnt // self.config.batch_size + 1
        self._training_step = step
        logging.info("[+] trainning step %d" % step)
        return step

    def _train_gen(self, once=False, title_or_body='title'):
        window_size = self.config.window_size

        docs_window = self.DocsWindow(self.docs, window_size)
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        all_pos = []
                        all_ims = []
                        if len(ih) > self.config.max_impression:
                            continue

                        for impression in ih:

                            if len(impression.pos) > self.config.max_impression_pos:
                                continue

                            if len(impression.neg) > self.config.max_impression_neg:
                                continue

                            for pos in impression.pos:
                                all_pos.append(pos)
                                all_ims.append(impression)

                        #                        sample_flag = False
                        #                        if max_user_clicked_count != -1 and len(all_pos)>max_user_clicked_count:
                        #                            sample_pos = set(np.random.choice(all_pos,size=max_user_clicked_count,replace=False))
                        #                            sample_flag = True

                        if len(all_pos) <= 1:
                            continue

                        i = -1
                        pos_len = len(all_pos) - 1

                        for pos in all_pos:
                            i = i + 1
                            #                                if sample_flag and pos not in sample_pos:
                            #                                    continue

                            seq = all_pos[:i] + all_pos[i + 1:]

                            if pos_len >= window_size:
                                seq = np.random.choice(seq, size=window_size, replace=False)
                            else:
                                seq = np.random.choice(seq, size=pos_len, replace=False)

                            if title_or_body == 'title':
                                clicked = docs_window.get_title(seq)
                                yield clicked, self.docs[pos].title, 1
                                for neg in all_ims[i].negative_samples(self.config.negative_samples):
                                    yield clicked, self.docs[neg].title, 0
                            elif title_or_body == 'body':
                                clicked = docs_window.get_body(seq)
                                yield clicked, self.docs[pos].body, 1
                                for neg in all_ims[i].negative_samples(self.config.negative_samples):
                                    yield clicked, self.docs[neg].body, 0

                            elif title_or_body == 'doc':
                                clicked = docs_window.get_doc(seq)
                                yield clicked, self.docs[pos].doc, 1
                                for neg in all_ims[i].negative_samples(self.config.negative_samples):
                                    yield clicked, self.docs[neg].doc, 0
                            elif title_or_body == 'concat':
                                clicked_title = docs_window.get_title(seq)
                                clicked_body = docs_window.get_body(seq)
                                clicked = np.concatenate([clicked_title, clicked_body], axis=-1)
                                candidate = np.concatenate([self.docs[pos].title, self.docs[pos].body], axis=-1)
                                yield clicked, candidate, 1
                                for neg in all_ims[i].negative_samples(self.config.negative_samples):
                                    candidate = np.concatenate([self.docs[neg].title, self.docs[neg].body], axis=-1)
                                    yield clicked, candidate, 0
                            elif title_or_body == 'title_body':
                                clicked_title = docs_window.get_title(seq)
                                clicked_body = docs_window.get_body(seq)

                                yield clicked_title, clicked_body, self.docs[pos].title, self.docs[pos].body, 1
                                for neg in all_ims[i].negative_samples(self.config.negative_samples):
                                    yield clicked_title, clicked_body, self.docs[neg].title, self.docs[neg].body, 0

            if once:
                break

    def train_gen(self, once=False):
        return self._train_gen(once, 'title')


class Seq2VecRandomProduct(Seq2VeRandom):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits


class Seq2VecEncoderProduct(Seq2VecEncoder):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits


class Seq2VecBody(Seq2VeRandom):
    def get_doc_encoder(self):
        return self._get_doc_encoder(self.config.body_shape)

    def _build_model(self):
        self.modeling(self.config.body_shape)

    def train_gen(self, once=False):
        return self._train_gen(once, 'body')

    def valid_gen(self):
        return self._valid_gen('body')

    def test_gen(self):
        return self._test_gen('body')


class Seq2VecBodyProduct(Seq2VecBody):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits


class Seq2VecTitleBodyConcat(Seq2VeRandom):
    def get_doc_encoder(self):
        return self._get_doc_encoder(self.config.title_shape + self.config.body_shape)

    def _build_model(self):
        self.modeling(self.config.title_shape + self.config.body_shape)

    def train_gen(self, once=False):
        return self._train_gen(once, 'concat')

    def valid_gen(self):
        return self._valid_gen('concat')

    def test_gen(self):
        return self._test_gen('concat')


class Seq2VecTitleBodyConcatProduct(Seq2VecTitleBodyConcat):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits


class Seq2VecDoc(Seq2VeRandom):
    class News:
        __slots__ = ['doc', ]

        def __init__(self, doc):
            self.doc = doc

    def _load_docs(self):
        logging.info("[+] loading docs metadata")
        doc_parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.title_shape + self.config.body_shape)
        )

        with utils.open(self.config.doc_meta_input) as file:
            docs = [line.strip('\n').split('\t') for line in file]

        self.docs = {
            int(line[1]): self.News(
                doc_parser(line[4] + ' ' + line[5])[0],
            ) for line in docs}

        self.doc_count = max(self.docs.keys()) + 1
        doc_example = self.docs[self.doc_count - 1]
        self.docs[0] = self.News(
            np.zeros_like(doc_example.doc))

        logging.info("[-] loaded docs metadata")

    def get_doc_encoder(self):
        return self._get_doc_encoder(self.config.title_shape + self.config.body_shape)

    def _build_model(self):
        self.modeling(self.config.title_shape + self.config.body_shape)

    def train_gen(self, once=False):
        return self._train_gen(once, 'doc')

    def valid_gen(self):
        return self._valid_gen('doc')

    def test_gen(self):
        return self._test_gen('doc')


class Seq2VecDocProduct(Seq2VecDoc):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits


class Doc2VecProduct(Seq2VecDoc):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits

    def test_gen1(self, doc2vec=None):
        def _gen_doc(clicked, impression):
            for p in impression.pos:
                vec = doc2vec[p]
                yield clicked, vec, 1
            for n in impression.neg:
                vec = doc2vec[n]
                yield clicked, vec, 0

        with open(self.config.training_data_input if self.is_training else self.config.testing_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                if line[2] and line[3]:
                    ih1 = self._extract_impressions(line[2])
                    ih2 = self._extract_impressions(line[3])
                    ch = self.Window(self.docs, self.config.test_window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        clicked = np.stack([doc2vec[i] for i in ch.click_history])
                        yield list(_gen_doc(clicked, impression))
                        for pos in impression.pos:
                            ch.push(pos)

    @property
    def test1(self):
        def _gen_doc2vec():
            batch_size = self.config.batch_size
            bc_inputs = []
            docs_idx = []
            cnt = 0
            doc2vec = {}
            doc_encoder = self.model.get_layer('doc_encoder')

            for idx, doc in self.docs.items():
                docs_idx.append(idx)

                bc_inputs.append(doc.doc)
                cnt = cnt + 1
                if cnt == batch_size:
                    outputs = doc_encoder.predict(np.stack(bc_inputs))

                    for idx, o in zip(docs_idx, outputs):
                        doc2vec[idx] = o

                    bc_inputs = []
                    docs_idx = []
                    cnt = 0

            if cnt != 0:
                outputs = doc_encoder.predict(np.stack(bc_inputs))

                for idx, o in zip(docs_idx, outputs):
                    doc2vec[idx] = o

                bc_inputs = []
                docs_idx = []
                cnt = 0
            doc2vec[0] = np.zeros_like(doc2vec[0])
            return doc2vec

        doc2vec = _gen_doc2vec()

        def _get_score_encoder():
            user_vec = keras.layers.Input((self.config.user_embedding_dim,))
            candidate_vec = keras.layers.Input((self.config.user_embedding_dim,))

            score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
            logits = keras.layers.Activation('sigmoid')(score)
            return keras.Model([user_vec, candidate_vec], logits)

        user_encoder = self.get_user_encoder(self.config.test_window_size)
        score_encoder = _get_score_encoder()

        for b in self.test_gen1(doc2vec):
            batch = [np.stack(x) for x in zip(*b)]

            user_vec = user_encoder.predict(batch[0])

            doc_vec = batch[1]
            pred = score_encoder.predict([user_vec, doc_vec]).reshape(-1)

            #            score = (user_vec * doc_vec).sum(axis=1)
            #            pred = 1./(1.+np.exp(-score))
            yield [pred, batch[-1]]


class Seq2VecTitleBody(Seq2VeRandom):
    def train_gen(self, once=False):
        return self._train_gen(once, 'title_body')

    def valid_gen(self):
        return self._valid_gen('title_body')

    def test_gen(self):
        return self._test_gen('title_body')

    def get_doc_encoder(self):
        if self.config.enable_pretrain_encoder:
            encoder = utils.load_model(self.config.encoder_input)
            if not self.config.pretrain_encoder_trainable:
                encoder.trainable = False
            return encoder
        else:
            if self.config.debug:
                title_embedding = np.load(self.config.title_embedding_input + '.npy')
            else:
                title_embedding = utils.load_textual_embedding(self.config.title_embedding_input,
                                                               self.config.textual_embedding_dim)

            news_model = self.config.news_encoder

            title_shape = self.config.title_shape
            body_shape = self.config.body_shape

            if news_model == 'cnnatt':
                mask_zero = False
            elif news_model == 'cnnattq':
                mask_zero = False
            else:
                raise Exception('Unsupport doc model')

            title_embedding_layer = keras.layers.Embedding(
                *title_embedding.shape,

                weights=[title_embedding],
                trainable=self.config.textual_embedding_trainable,
                mask_zero=mask_zero
            )

            if news_model == 'cnnatt':
                inp_title = keras.Input((title_shape,), dtype='int32')
                emb_title = title_embedding_layer(inp_title)

                inp_body = keras.Input((body_shape,), dtype='int32')
                emb_body = title_embedding_layer(inp_body)

                filter_count, filter_size = self.config.title_filter_shape

                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

                e_title = keras.layers.Dropout(self.config.dropout)(emb_title)
                c_title = cnn(e_title)

                cmask_title = keras.layers.Masking()(c_title)
                a_title = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask_title))

                e_body = keras.layers.Dropout(self.config.dropout)(emb_body)
                c_body = cnn(e_body)

                cmask_body = keras.layers.Masking()(c_body)
                a_body = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask_body))

                concat_layer = keras.layers.Concatenate()
                doc = concat_layer([a_title, a_body])

                dense_layer = keras.layers.Dense(self.config.user_embedding_dim)
                vec = dense_layer(doc)

                title_encoder = keras.Model(inp_title, a_title)
                body_encoder = keras.Model(inp_body, a_body)
                doc_encoder = keras.Model([inp_title, inp_body], vec, name='doc_encoder')

            elif news_model == 'cnnattq':
                inp_title = keras.Input((title_shape,), dtype='int32')
                emb_title = title_embedding_layer(inp_title)

                inp_body = keras.Input((body_shape,), dtype='int32')
                emb_body = title_embedding_layer(inp_body)

                filter_count, filter_size = self.config.title_filter_shape

                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

                e_title = keras.layers.Dropout(self.config.dropout)(emb_title)
                c_title = cnn(e_title)

                att_layer = models.SimpleAttentionMaskSupport()

                cmask_title = keras.layers.Masking()(c_title)
                a_title = att_layer(keras.layers.Dropout(self.config.dropout)(cmask_title))

                e_body = keras.layers.Dropout(self.config.dropout)(emb_body)
                c_body = cnn(e_body)

                cmask_body = keras.layers.Masking()(c_body)
                a_body = att_layer(keras.layers.Dropout(self.config.dropout)(cmask_body))

                concat_layer = keras.layers.Concatenate()
                doc = concat_layer([a_title, a_body])

                dense_layer = keras.layers.Dense(self.config.user_embedding_dim)
                vec = dense_layer(doc)

                title_encoder = keras.Model(inp_title, a_title)
                body_encoder = keras.Model(inp_body, a_body)
                doc_encoder = keras.Model([inp_title, inp_body], vec, name='doc_encoder')
            else:
                raise Exception('Unsupport doc model')

            return doc_encoder, title_encoder, body_encoder, concat_layer, dense_layer

    def _build_model(self):
        title_shape = self.config.title_shape
        body_shape = self.config.body_shape
        inp_title = keras.Input((self.config.window_size, title_shape), dtype='int32')
        inp_body = keras.Input((self.config.window_size, body_shape,), dtype='int32')

        doc_encoder, title_encoder, body_encoder, concat_layer, dense_layer = self.get_doc_encoder()

        user_title_encoder_dist = keras.layers.TimeDistributed(title_encoder)
        user_body_encoder_dist = keras.layers.TimeDistributed(body_encoder)

        title_vec = user_title_encoder_dist(inp_title)
        body_vec = user_body_encoder_dist(inp_body)
        concat_vec = concat_layer([title_vec, body_vec])
        clicked_vec_raw = dense_layer(concat_vec)

        mask = models.ComputeMasking(0)(inp_title)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])

        user_encoder = self.get_user_encoder()

        user_vec = user_encoder(clicked_vec)

        candidate_title = keras.Input((title_shape,))
        candidate_body = keras.Input((body_shape,))

        candidate_vec = doc_encoder([candidate_title, candidate_body])

        logits = self.score_encoder(user_vec, candidate_vec)

        self.model = keras.Model([inp_title, inp_body, candidate_title, candidate_body], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )


class Seq2VecTitleBodyProduct(Seq2VecTitleBody):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits


class Seq2VecPersonal(Seq2VecEncoder):
    def _load_users(self):
        super(Seq2VecPersonal, self)._load_users()
        count = 0
        self.user_mapping = {}
        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                self.user_mapping[line[0] + line[1]] = len(self.user_mapping)
                count += 1

        self.user_count = count

    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:

                for line in file:
                    line = line.strip('\n').split('\t')

                    user_idx = self.user_mapping[line[0] + line[1]]

                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                ch.push(pos)
                        for impression in ih:
                            for pos in impression.pos:
                                clicked = ch.get_title([pos])
                                yield user_idx, clicked, self.docs[pos].title, 1

                                for neg in impression.negative_samples(self.config.negative_samples):
                                    yield user_idx, clicked, self.docs[neg].title, 0

    def valid_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')

                    user_idx = self.user_mapping[line[0] + line[1]]
                    if line[2] and line[3]:
                        ih1 = self._extract_impressions(line[2])
                        ih2 = self._extract_impressions(line[3])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih1:
                            for pos in impression.pos:
                                ch.push(pos)
                        for impression in ih2:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield user_idx, clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user_idx, clicked, self.docs[neg].title, 0
                                ch.push(pos)

    def test_gen(self):
        def __gen__(_user_idx, _clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user_idx, _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user_idx, _clicked, doc.title, 0

        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                user_idx = self.user_mapping[line[0] + line[1]]
                if line[2] and line[3]:
                    ih1 = self._extract_impressions(line[2])
                    ih2 = self._extract_impressions(line[3])
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        clicked = ch.get_title()
                        yield list(__gen__(user_idx, clicked, impression))
                        for pos in impression.pos:
                            ch.push(pos)

    @property
    def test(self):
        for b in self.test_gen():
            batch = [np.stack(x) for x in zip(*b)]
            yield [self.model.predict([batch[0], batch[1], batch[2]]).reshape(-1), batch[3]]

    def get_user_encoder(self):

        user_idx = keras.Input((1,))

        user_embedding_layer = keras.layers.Embedding(
            self.user_count,
            self.config.personal_embedding_dim,
            trainable=True)

        user_emb = keras.layers.Reshape((-1,))(user_embedding_layer(user_idx))
        query = keras.layers.Dense(self.config.user_embedding_dim, activation='relu')(user_emb)

        user_clicked_vec = keras.Input((self.config.window_size, self.config.user_embedding_dim),
                                       name='user_clicked_vec')

        user_model = self.config.arch

        if user_model == 'sqatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = models.SimpleQueryAttentionMasked()([mask_clicked_vec, query])
        elif user_model == 'nsqatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            query = keras.layers.BatchNormalization()(query)
            user_vec = models.SimpleQueryAttentionMasked()([mask_clicked_vec, query])
        elif user_model == 'esqatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            query = user_emb
            user_vec = models.SimpleQueryAttentionMasked()([mask_clicked_vec, query])
        #        elif user_model == 'qatt':
        #
        #            query = user_emb
        #            user_vec = models.QueryAttentionMasked(mask)([clicked_vec,query])
        elif user_model == 'dsqatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            query = user_emb
            mask_clicked_vec = keras.layers.Dense(self.config.personal_embedding_dim,
                                                  activation=keras.activations.tanh)(mask_clicked_vec)
            user_vec = models.SimpleQueryAttentionMasked()([mask_clicked_vec, query])
        elif user_model == 'deqatt':

            query = user_emb
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = models.TanhQueryAttentionMasked(self.config.personal_embedding_dim)([mask_clicked_vec, query])
        elif user_model == 'dqatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = models.TanhQueryAttentionMasked(self.config.user_embedding_dim)([mask_clicked_vec, query])

        elif user_model == 'gruqatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            mask_clicked_vec = keras.layers.GRU(self.config.user_embedding_dim, return_sequences=True)(mask_clicked_vec)
            user_vec = models.TanhQueryAttentionMasked(self.config.user_embedding_dim)([mask_clicked_vec, query])
        elif user_model == 'grusqatt':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            mask_clicked_vec = keras.layers.GRU(self.config.user_embedding_dim, return_sequences=True)(mask_clicked_vec)
            user_vec = models.SimpleQueryAttentionMasked()([mask_clicked_vec, query])
        else:
            raise Exception('Unsupport user model')

        user_encoder = keras.Model([user_clicked_vec, user_idx], user_vec, name='user_encoder')

        return user_encoder

    def _build_model(self):
        user_idx = keras.Input((1,))

        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])

        user_encoder = self.get_user_encoder()
        user_vec = user_encoder([clicked_vec, user_idx])

        candidate = keras.Input((self.config.title_shape,))
        candidate_vec = doc_encoder(candidate)

        logits = self.score_encoder(user_vec, candidate_vec)

        self.model = keras.Model([user_idx, clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )


class Seq2VecPersonalShort(Seq2VecPersonal):
    def _build_model(self):
        super(Seq2VecPersonalShort, self)._build_model()
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )

    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    user_idx = self.user_mapping[line[0] + line[1]]

                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield user_idx, clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user_idx, clicked, self.docs[neg].title, 0
                                ch.push(pos)


class Seq2VecLongOrder(Seq2VecEncoder):
    class DocsWindow:
        def __init__(self, docs, window_size):

            self.docs = docs
            self.window_size = window_size

        def get_title(self, seq):
            if len(seq) < self.window_size:
                seq = np.pad(seq, (self.window_size - len(seq), 0), mode='constant')

            return np.stack([self.docs[i].title for i in seq])

        def get_body(self, seq):
            if len(seq) < self.window_size:
                seq = np.pad(seq, (self.window_size - len(seq), 0), mode='constant')

            return np.stack([self.docs[i].body for i in seq])

    def train_gen(self):
        window_size = self.config.window_size

        docs_window = self.DocsWindow(self.docs, window_size)
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        all_pos = []

                        for impression in ih:
                            for pos in impression.pos:
                                all_pos.append(pos)

                        i = 0
                        for impression in ih:
                            for pos in impression.pos:
                                if i >= window_size:
                                    seq = all_pos[(i - window_size):i]
                                else:
                                    seq1 = all_pos[:i]
                                    seq2 = all_pos[i + 1:window_size + 1]
                                    seq = seq1 + seq2

                                i = i + 1
                                clicked = docs_window.get_title(seq)

                                yield clicked, self.docs[pos].title, 1
                                for neg in impression.negative_samples(self.config.negative_samples):
                                    yield clicked, self.docs[neg].title, 0


class Seq2VecEncoderShort(Seq2VecEncoder):
    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    user_idx = self.user_mapping[line[0] + line[1]]

                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield user_idx, clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user_idx, clicked, self.docs[neg].title, 0
                                ch.push(pos)


class Seq2VecBodySent(Seq2VecEncoder):
    def _load_docs(self):
        logging.info("[+] loading docs metadata")
        title_parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.title_shape)
        )

        split_tokens = []
        with utils.open(self.config.doc_punc_index_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                if line[0] == '.' or line[0] == '?' or line[0] == '!':
                    split_tokens.append(int(line[1]))

        body_parser = document.DocumentParser(
            document.parse_document(),
            document.clause(split_tokens),
            document.pad_docs(self.config.body_sent_cnt, self.config.body_sent_len)
        )
        with utils.open(self.config.doc_meta_input) as file:
            docs = [line.strip('\n').split('\t') for line in file]

        self.docs = {
            int(line[1]): self.News(
                title_parser(line[4])[0],
                body_parser(line[5])[0],
            ) for line in docs}

        self.doc_count = max(self.docs.keys()) + 1
        doc_example = self.docs[self.doc_count - 1]
        self.docs[0] = self.News(
            np.zeros_like(doc_example.title),
            np.zeros_like(doc_example.body))

        logging.info("[-] loaded docs metadata")

    def get_doc_encoder(self):
        if self.config.enable_pretrain_encoder:
            encoder = utils.load_model(self.config.encoder_input)
            if not self.config.pretrain_encoder_trainable:
                encoder.trainable = False
            return encoder
        else:
            if self.config.debug:
                embedding_weight = np.load(self.config.title_embedding_input + '.npy')
            else:
                embedding_weight = utils.load_textual_embedding(self.config.title_embedding_input,
                                                                self.config.textual_embedding_dim)

            news_model = self.config.news_encoder

            if news_model == 'caca':
                mask_zero = False
            else:
                raise Exception('Unsupport doc model')

            embedding_layer = keras.layers.Embedding(
                *embedding_weight.shape,
                input_length=self.config.body_sent_len,
                weights=[embedding_weight],
                trainable=self.config.textual_embedding_trainable,
                mask_zero=mask_zero
            )

            input_count = self.config.body_sent_cnt
            input_length = self.config.body_sent_len

            filter_count, filter_size = self.config.body_filter_shape
            output_dim = self.config.user_embedding_dim
            dropout = self.config.dropout
            if news_model == 'caca':

                line_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

                line_i = keras.Input((input_length,), dtype='int32')
                line_e = embedding_layer(line_i)
                line_c = line_cnn(line_e)
                line_a = models.simple_attention(line_c)

                encoder = keras.layers.TimeDistributed(keras.Model(line_i, line_a))
                global_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu',
                                                 strides=1)

                global_i = keras.Input((input_count, input_length), dtype='int32')
                global_e = encoder(global_i)
                global_c = global_cnn(global_e)
                global_a = models.simple_attention(global_c)

                out = keras.layers.Dense(output_dim)(global_a)
                doc_encoder = keras.Model(global_i, out, name='doc_encoder')
            elif news_model == 'cacadp1':
                line_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

                line_i = keras.Input((input_length,), dtype='int32')
                line_e = keras.layers.Dropout(dropout)(embedding_layer(line_i))
                line_c = line_cnn(line_e)
                line_a = models.simple_attention(keras.layers.Dropout(dropout)(line_c))

                encoder = keras.layers.TimeDistributed(keras.Model(line_i, line_a))
                global_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu',
                                                 strides=1)

                global_i = keras.Input((input_count, input_length), dtype='int32')
                global_e = encoder(global_i)
                global_c = global_cnn(global_e)
                global_a = models.simple_attention(keras.layers.Dropout(dropout)(global_c))

                out = keras.layers.Dense(output_dim)(global_a)
                doc_encoder = keras.Model(global_i, out, name='doc_encoder')
            elif news_model == 'cacabn':
                line_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

                line_i = keras.Input((input_length,), dtype='int32')
                line_e = embedding_layer(line_i)
                line_c = line_cnn(line_e)
                line_a = models.simple_attention(line_c)

                encoder = keras.layers.TimeDistributed(keras.Model(line_i, line_a))
                global_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu',
                                                 strides=1)

                global_i = keras.Input((input_count, input_length), dtype='int32')
                global_e = encoder(global_i)
                global_c = global_cnn(global_e)
                global_a = models.simple_attention(global_c)
                global_a = keras.layers.BatchNormalization()(global_a)
                out = keras.layers.Dense(output_dim)(global_a)
                doc_encoder = keras.Model(global_i, out, name='doc_encoder')
            else:
                raise Exception('Unsupport doc model')

            return doc_encoder

    def _build_model(self):
        clicked = keras.Input((self.config.window_size, self.config.body_sent_cnt, self.config.body_sent_len))
        doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(keras.layers.Reshape((self.config.window_size, -1))(clicked))
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])

        user_encoder = self.get_user_encoder()

        user_vec = user_encoder(clicked_vec)

        candidate = keras.Input((self.config.body_sent_cnt, self.config.body_sent_len))
        candidate_vec = doc_encoder(candidate)

        logits = self.score_encoder(user_vec, candidate_vec)

        self.model = keras.Model([clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )

    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                ch.push(pos)
                        for impression in ih:
                            for pos in impression.pos:
                                clicked = ch.get_body([pos])
                                yield clicked, self.docs[pos].body, 1
                                for neg in impression.negative_samples(self.config.negative_samples):
                                    yield clicked, self.docs[neg].body, 0

    def valid_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2] and line[3]:
                        ih1 = self._extract_impressions(line[2])
                        ih2 = self._extract_impressions(line[3])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih1:
                            for pos in impression.pos:
                                ch.push(pos)
                        for impression in ih2:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_body()
                                    yield clicked, self.docs[pos].body, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield clicked, self.docs[neg].body, 0
                                ch.push(pos)

    def test_gen(self):
        def __gen__(_clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _clicked, doc.body, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _clicked, doc.body, 0

        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                if line[2] and line[3]:
                    ih1 = self._extract_impressions(line[2])
                    ih2 = self._extract_impressions(line[3])
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        clicked = ch.get_body()
                        yield list(__gen__(clicked, impression))
                        for pos in impression.pos:
                            ch.push(pos)
