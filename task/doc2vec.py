# -*- coding: utf-8 -*-

from .seq2vec import *
import keras
import utils
import numpy as np
import logging
import document
import models
import tensorflow as tf
from sklearn.metrics import roc_auc_score


class Doc2Vec(Seq2VecLongEncoder):
    class Window:
        __slots__ = ['count', 'docs', 'click_history']

        def __init__(self, docs, window_size):
            self.count = 0
            self.docs = docs
            self.click_history = [0 for _ in range(window_size)]

        def get_title(self, elimination=None):
            if elimination:
                return np.stack(
                    [self.docs[i].title if i not in elimination else self.docs[0].title for i in self.click_history])
            else:
                return np.stack([self.docs[i].title for i in self.click_history])

        def get_body(self, elimination=None):
            if elimination:
                return np.stack(
                    [self.docs[i].body if i not in elimination else self.docs[0].body for i in self.click_history])
            else:
                return np.stack([self.docs[i].body for i in self.click_history])

        def get_doc(self, elimination=None):
            if elimination:
                return np.stack(
                    [self.docs[i].doc if i not in elimination else self.docs[0].doc for i in self.click_history])
            else:
                return np.stack([self.docs[i].doc for i in self.click_history])

        def get_vertical(self, elimination=None):
            if elimination:
                return np.stack(
                    [self.docs[i].vertical if i not in elimination else self.docs[0].vertical for i in
                     self.click_history])
            else:
                return np.stack([self.docs[i].vertical for i in self.click_history])

        def get_subvertical(self, elimination=None):
            if elimination:
                return np.stack(
                    [self.docs[i].subvertical if i not in elimination else self.docs[0].subvertical for i in
                     self.click_history])
            else:
                return np.stack([self.docs[i].subvertical for i in self.click_history])

        def push(self, doc):
            self.click_history.append(doc)
            self.click_history.pop(0)
            self.count += 1

        @property
        def window_size(self):
            return len(self.click_history)

    class News:

        def __init__(self, title, body, doc, vertical, subvertical):
            self.title = title
            self.body = body
            self.doc = doc
            self.vertical = vertical
            self.subvertical = subvertical

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

        def get_vertical(self, seq):
            if len(seq) < self.window_size:
                seq = np.pad(seq, (self.window_size - len(seq), 0), mode='constant')

            return np.stack([self.docs[i].vertical for i in seq])

        def get_subvertical(self, seq):
            if len(seq) < self.window_size:
                seq = np.pad(seq, (self.window_size - len(seq), 0), mode='constant')

            return np.stack([self.docs[i].subvertical for i in seq])

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

        doc_parser = document.DocumentParser(
            document.parse_document(),
            document.pad_document(1, self.config.title_shape + self.config.body_shape)
        )

        vert2idx = {}
        vert_cnt = 1

        subvert2idx = {}
        subvert_cnt = 1
        vert2idx_path = self.config.vertical2idx_input
        with utils.open(vert2idx_path[0]) as file:
            for line in file:
                vert, idx = line.strip('\n').split('\t')
                vert2idx[vert] = idx
                vert_cnt = vert_cnt + 1
        with utils.open(vert2idx_path[1]) as file:
            for line in file:
                vert, idx = line.strip('\n').split('\t')
                subvert2idx[vert] = idx
                subvert_cnt = subvert_cnt + 1

        self.vert_cnt = vert_cnt
        self.subvert_cnt = subvert_cnt
        with utils.open(self.config.doc_meta_input) as file:
            docs = [line.strip('\n').split('\t') for line in file]

        self.docs = {
            int(line[1]): self.News(
                title_parser(line[4])[0],
                body_parser(line[5])[0],
                doc_parser(line[4] + ' ' + line[5])[0],
                [int(vert2idx[line[2]])],
                [int(subvert2idx[line[3]])]
            ) for line in docs}

        self.doc_count = max(self.docs.keys()) + 1
        doc_example = self.docs[self.doc_count - 1]
        self.docs[0] = self.News(
            np.zeros_like(doc_example.title),
            np.zeros_like(doc_example.body),
            np.zeros_like(doc_example.doc),
            [0], [0])

        logging.info("[-] loaded docs metadata")

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

    def _train_gen(self, once=False):
        window_size = self.config.window_size
        docs_window = self.DocsWindow(self.docs, window_size)
        while True:
            with utils.open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        all_pos = []
                        all_ims = []
                        if self.config.max_impression != -1 and len(ih) > self.config.max_impression:
                            continue

                        for impression in ih:

                            if self.config.max_impression_pos != -1 and len(
                                    impression.pos) > self.config.max_impression_pos:
                                continue

                            if self.config.max_impression_neg != -1 and len(
                                    impression.neg) > self.config.max_impression_neg:
                                continue

                            for pos in impression.pos:
                                all_pos.append(pos)
                                all_ims.append(impression)

                        if len(all_pos) <= 1:
                            continue

                        i = -1
                        pos_len = len(all_pos) - 1

                        for pos in all_pos:
                            i = i + 1
                            seq = all_pos[:i] + all_pos[i + 1:]

                            if pos_len >= window_size:
                                seq = np.random.choice(seq, size=window_size, replace=False)
                            else:
                                seq = np.random.choice(seq, size=pos_len, replace=False)

                            for sam in self.train_sample(docs_window, seq, pos,
                                                         all_ims[i].negative_samples(self.config.negative_samples)):
                                yield sam
            if once:
                break

    def train_gen(self, once=False):
        return self._train_gen(once)

    def train_sample(self, docs_window, clicked_docs, pos, negs):
        clicked = docs_window.get_doc(clicked_docs)
        clicked_vertical = docs_window.get_vertical(clicked_docs)
        clicked_subvertical = docs_window.get_subvertical(clicked_docs)
        clicked = np.concatenate([clicked_vertical, clicked_subvertical, clicked], axis=-1)

        yield clicked, np.concatenate([self.docs[pos].vertical, self.docs[pos].subvertical, self.docs[pos].doc],
                                      axis=-1), 1
        for neg in negs:
            yield clicked, np.concatenate([self.docs[neg].vertical, self.docs[neg].subvertical, self.docs[neg].doc],
                                          axis=-1), 0

    def valid_gen(self):
        return self._valid_gen()

    def _valid_gen(self):
        while True:
            with utils.open(self.config.training_data_input) as file:
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
                                    for sam in self.valid_sample(ch, pos, impression.negative_samples(
                                            self.config.negative_samples)):
                                        yield sam

                                ch.push(pos)

    def valid_sample(self, ch, pos, negs):
        clicked = ch.get_doc()
        clicked_vertical = ch.get_vertical()
        clicked_subvertical = ch.get_subvertical()
        clicked = np.concatenate([clicked_vertical, clicked_subvertical, clicked], axis=-1)

        yield clicked, np.concatenate([self.docs[pos].vertical, self.docs[pos].subvertical, self.docs[pos].doc],
                                      axis=-1), 1
        for neg in negs:
            yield clicked, np.concatenate([self.docs[neg].vertical, self.docs[neg].subvertical, self.docs[neg].doc],
                                          axis=-1), 0

    def callback(self, epoch):
        metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
        keras.backend.get_session().run(tf.initializers.variables(metric_vars))

        keras.backend.set_value(self.model.optimizer.lr, keras.backend.get_value(self.model.optimizer.lr) * 0.2)

        if epoch or True:
            def __gen__():
                if self.config.testing_impression == -1:
                    for i, (y_pred, y_true) in enumerate(self.test):
                        auc = roc_auc_score(y_true, y_pred)
                        ndcgx = utils.ndcg_score(y_true, y_pred, 10)
                        ndcgv = utils.ndcg_score(y_true, y_pred, 5)
                        mrr = utils.mrr_score(y_true, y_pred)
                        pos = np.sum(y_true)
                        size = len(y_true)
                        yield auc, ndcgx, ndcgv, mrr, pos, size, i
                else:
                    for i, (y_pred, y_true) in zip(range(self.config.testing_impression), self.test):
                        auc = roc_auc_score(y_true, y_pred)
                        ndcgx = utils.ndcg_score(y_true, y_pred, 10)
                        ndcgv = utils.ndcg_score(y_true, y_pred, 5)
                        mrr = utils.mrr_score(y_true, y_pred)
                        pos = np.sum(y_true)
                        size = len(y_true)
                        yield auc, ndcgx, ndcgv, mrr, pos, size, i

            values = [np.mean(x) for x in zip(*__gen__())]
            utils.logging_evaluation(dict(auc=values[0], ndcgx=values[1], ndcgv=values[2], mrr=values[3]))
            utils.logging_evaluation(dict(pos=values[4], size=values[5], num=values[6] * 2 + 1))

            if epoch == self.config.epochs - 1 and tf.gfile.Exists(self.config.testing_data_input):
                self.is_training = False
                values = [np.mean(x) for x in zip(*__gen__())]
                utils.logging_evaluation(dict(auc=values[0], ndcgx=values[1], ndcgv=values[2], mrr=values[3]))
                utils.logging_evaluation(dict(pos=values[4], size=values[5], num=values[6] * 2 + 1))

    def get_test_user_encoder(self):
        return self.get_user_encoder(self.config.test_window_size)

    def test_doc_sample(self, doc):
        return np.concatenate([doc.vertical, doc.subvertical, doc.doc], axis=-1)

    def test_gen(self, doc2vec=None):
        def _gen_doc(clicked, impression):
            for p in impression.pos:
                vec = doc2vec[p]
                yield clicked, vec, 1
            for n in impression.neg:
                vec = doc2vec[n]
                yield clicked, vec, 0

        with utils.open(
                self.config.training_data_input if self.is_training else self.config.testing_data_input) as file:
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
    def test(self):
        def _gen_doc2vec():
            batch_size = self.config.batch_size
            bc_inputs = []
            docs_idx = []
            cnt = 0
            doc2vec = {}
            doc_encoder = self.model.get_layer('doc_encoder')

            for idx, doc in self.docs.items():
                docs_idx.append(idx)

                bc_inputs.append(self.test_doc_sample(doc))
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

        user_encoder = self.get_test_user_encoder()
        score_encoder = self.model.get_layer('score_encoder')

        for b in self.test_gen(doc2vec):
            batch = [np.stack(x) for x in zip(*b)]

            user_vec = user_encoder.predict(batch[0])

            doc_vec = batch[1]
            score = score_encoder.predict([user_vec, doc_vec]).reshape(-1)

            pred = 1. / (1. + np.exp(-score))
            yield [pred, batch[-1]]

    def get_doc_encoder(self):
        return self._get_doc_encoder(self.config.title_shape + self.config.body_shape)

    def _get_doc_encoder(self, input_shape=None):
        if self.config.enable_pretrain_encoder:
            encoder = utils.load_model(self.config.encoder_input)
            if not self.config.pretrain_encoder_trainable:
                encoder.trainable = False
            return encoder
        else:
            if self.config.debug:
                embed_wei = np.load(self.config.title_embedding_input + '.npy')
            else:
                embed_wei = utils.load_textual_embedding(self.config.title_embedding_input,
                                                         self.config.textual_embedding_dim)

            news_model = self.config.news_encoder

            inp_shape = input_shape

            if news_model == 'cnnatt':
                mask_zero = False
                vert_shape = (self.vert_cnt, self.config.textual_embedding_dim)
                subvert_shape = (self.subvert_cnt, self.config.textual_embedding_dim)
            elif news_model == 'vert-ca-fc':
                mask_zero = False
                vert_shape = (self.vert_cnt, self.config.vertical_embedding_dim)
                subvert_shape = (self.subvert_cnt, self.config.subvertical_embedding_dim)
            elif news_model == 'vert-fc-ca-fc':
                mask_zero = False
                vert_shape = (self.vert_cnt, self.config.vertical_embedding_dim)
                subvert_shape = (self.subvert_cnt, self.config.subvertical_embedding_dim)
            elif news_model == 'vert-tanh-ca':
                mask_zero = False
                vert_shape = (self.vert_cnt, self.config.vertical_embedding_dim)
                subvert_shape = (self.subvert_cnt, self.config.subvertical_embedding_dim)
            elif news_model == 'qcnnatt':
                mask_zero = False
                vert_shape = (self.vert_cnt, self.config.vertical_embedding_dim)
                subvert_shape = (self.subvert_cnt, self.config.subvertical_embedding_dim)
            else:
                raise Exception('Unsupport doc model')

            title_embedding_layer = keras.layers.Embedding(
                *embed_wei.shape,
                input_length=inp_shape,
                weights=[embed_wei],
                trainable=self.config.textual_embedding_trainable,
                mask_zero=mask_zero
            )

            vert_embedding_layer = keras.layers.Embedding(
                *vert_shape,
                input_length=1,
                trainable=True)

            subvert_embedding_layer = keras.layers.Embedding(
                *subvert_shape,
                input_length=1,
                trainable=True)

            inp = keras.Input((2 + inp_shape,), dtype='int32')
            vert_in = models.SliceAxis1((0, 1))(inp)
            subvert_in = models.SliceAxis1((1, 2))(inp)
            doc_inp = models.SliceAxis1((2, 2 + inp_shape))(inp)
            emb = title_embedding_layer(doc_inp)
            vert_emb = vert_embedding_layer(vert_in)
            subvert_emb = subvert_embedding_layer(subvert_in)

            if news_model == 'cnnatt':

                emb = keras.layers.Concatenate(axis=1)([vert_emb, subvert_emb, emb])

                filter_count, filter_size = self.config.title_filter_shape

                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

                emb = keras.layers.Dropout(self.config.dropout)(emb)
                c = cnn(emb)

                mask = models.ComputeMasking()(emb)
                c = models.OverwriteMasking()([c, mask])
                cmask = keras.layers.Masking()(c)
                a = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask))

                a = keras.layers.Dense(self.config.user_embedding_dim)(a)

                doc_encoder = keras.Model(inp, a, name='doc_encoder')
            elif news_model == 'vert-ca-fc':
                filter_count, filter_size = self.config.title_filter_shape
                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)
                emb = keras.layers.Dropout(self.config.dropout)(emb)
                c = cnn(emb)
                mask = models.ComputeMasking()(emb)
                c = models.OverwriteMasking()([c, mask])

                cmask = keras.layers.Masking()(c)
                a = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask))
                vert_emb = keras.layers.Flatten()(vert_emb)
                subvert_emb = keras.layers.Flatten()(subvert_emb)

                con = keras.layers.Concatenate(axis=-1)([vert_emb, subvert_emb, a])

                a = keras.layers.Dense(self.config.user_embedding_dim)(con)

                doc_encoder = keras.Model(inp, a, name='doc_encoder')
            elif news_model == 'vert-fc-ca-fc':
                filter_count, filter_size = self.config.title_filter_shape
                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)
                emb = keras.layers.Dropout(self.config.dropout)(emb)
                c = cnn(emb)

                mask = models.ComputeMasking()(emb)
                c = models.OverwriteMasking()([c, mask])
                cmask = keras.layers.Masking()(c)
                a = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask))

                vert_fc_dim = self.config.vertical_embedding_dim + self.config.subvertical_embedding_dim
                a = keras.layers.Dense(self.config.user_embedding_dim - vert_fc_dim)(a)

                vert_emb = keras.layers.Flatten()(vert_emb)
                subvert_emb = keras.layers.Flatten()(subvert_emb)

                con = keras.layers.Concatenate(axis=-1)([vert_emb, subvert_emb])
                con = keras.layers.Dropout(self.config.dropout)(con)
                con = keras.layers.Dense(vert_fc_dim)(con)

                a = keras.layers.Concatenate(axis=-1)([con, a])

                doc_encoder = keras.Model(inp, a, name='doc_encoder')
            elif news_model == 'vert-tanh-ca':
                filter_count, filter_size = self.config.title_filter_shape
                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)
                emb = keras.layers.Dropout(self.config.dropout)(emb)
                c = cnn(emb)

                mask = models.ComputeMasking()(emb)
                c = models.OverwriteMasking()([c, mask])
                cmask = keras.layers.Masking()(c)
                a = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask))

                vert_emb = keras.layers.Flatten()(vert_emb)
                subvert_emb = keras.layers.Flatten()(subvert_emb)

                con = keras.layers.Concatenate(axis=-1)([vert_emb, subvert_emb])

                con = keras.layers.Dense(filter_count, activation='tanh')(con)

                a = keras.layers.Multiply()([a, con])

                a = keras.layers.Dense(self.config.user_embedding_dim)(a)

                doc_encoder = keras.Model(inp, a, name='doc_encoder')
            elif news_model == 'qcnnatt':
                filter_count, filter_size = self.config.title_filter_shape
                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)
                emb = keras.layers.Dropout(self.config.dropout)(emb)
                c = cnn(emb)

                mask = models.ComputeMasking()(emb)
                c = models.OverwriteMasking()([c, mask])
                cmask = keras.layers.Masking()(c)
                #                a = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask))

                vert_emb = keras.layers.Flatten()(vert_emb)
                subvert_emb = keras.layers.Flatten()(subvert_emb)

                con = keras.layers.Concatenate(axis=-1)([vert_emb, subvert_emb])

                con = keras.layers.Dense(filter_count, activation='tanh')(con)

                a = models.SimpleQueryAttentionMasked()([cmask, con])

                a = keras.layers.Dense(self.config.user_embedding_dim)(a)

                doc_encoder = keras.Model(inp, a, name='doc_encoder')

            else:
                raise Exception('Unsupport doc model')
            doc_encoder.summary()
            return doc_encoder

    def get_user_encoder(self, window_size=None):
        if window_size is None:
            window_size = self.config.window_size

        user_clicked_vec = keras.Input((window_size, self.config.user_embedding_dim), name='user_clicked_vec')

        user_model = self.config.arch

        if user_model == 'avg':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = models.GlobalAveragePoolingMaskSupport()(mask_clicked_vec)
        else:
            raise Exception('Unsupport user model')

        user_encoder = keras.Model(user_clicked_vec, user_vec, name='user_encoder')
        user_encoder.summary()
        return user_encoder

    def get_score_encoder(self):
        user_vec = keras.layers.Input((self.config.user_embedding_dim,))
        candidate_vec = keras.layers.Input((self.config.user_embedding_dim,))

        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])

        return keras.Model([user_vec, candidate_vec], score, name='score_encoder')

    def modeling(self, input_shape):
        inp_shape = input_shape

        clicked = keras.Input((self.config.window_size, inp_shape))
        doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = models.OverwriteMasking()([clicked_vec_raw, mask])

        user_encoder = self.get_user_encoder()

        user_vec = user_encoder(clicked_vec)

        candidate = keras.Input((inp_shape,))
        candidate_vec = doc_encoder(candidate)

        score_encoder = self.get_score_encoder()

        score = score_encoder([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        self.model = keras.Model([clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )

        self.model.summary()

    def get_model(self):
        self.modeling(2 + self.config.title_shape + self.config.body_shape)

    def _build_model(self):
        self.get_model()


class Title2Vec(Doc2Vec):

    def train_sample(self, docs_window, clicked_docs, pos, negs):
        clicked = docs_window.get_title(clicked_docs)
        clicked_vertical = docs_window.get_vertical(clicked_docs)
        clicked_subvertical = docs_window.get_subvertical(clicked_docs)
        clicked = np.concatenate([clicked_vertical, clicked_subvertical, clicked], axis=-1)

        yield clicked, np.concatenate([self.docs[pos].vertical, self.docs[pos].subvertical, self.docs[pos].title],
                                      axis=-1), 1
        for neg in negs:
            yield clicked, np.concatenate([self.docs[neg].vertical, self.docs[neg].subvertical, self.docs[neg].title],
                                          axis=-1), 0

    def valid_sample(self, ch, pos, negs):
        clicked = ch.get_title()
        clicked_vertical = ch.get_vertical()
        clicked_subvertical = ch.get_subvertical()
        clicked = np.concatenate([clicked_vertical, clicked_subvertical, clicked], axis=-1)

        yield clicked, np.concatenate([self.docs[pos].vertical, self.docs[pos].subvertical, self.docs[pos].title],
                                      axis=-1), 1
        for neg in negs:
            yield clicked, np.concatenate([self.docs[neg].vertical, self.docs[neg].subvertical, self.docs[neg].title],
                                          axis=-1), 0

    def test_doc_sample(self, doc):
        return np.concatenate([doc.vertical, doc.subvertical, doc.title], axis=-1)

    def get_doc_encoder(self):
        return self._get_doc_encoder(self.config.title_shape)

    def get_model(self):
        self.modeling(2 + self.config.title_shape)
