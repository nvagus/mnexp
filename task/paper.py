from .seq2vec import *
import random
from datetime import datetime, timedelta


class Seq2VecPaper(Seq2VecForward):
    class Impression:
        __slots__ = ['pos', 'neg', 'time']

        def __init__(self, d):
            d = d.split('#TAB#')
            self.pos = [int(k) for k in d[0].split(' ')]
            self.neg = [int(k) for k in d[1].split(' ')]
            if len(d) >= 3:
                self.time = datetime.strptime(d[2], '%m/%d/%Y %I:%M:%S %p')

        def negative_samples(self, n):
            return np.random.choice(self.neg, n)

    def _extract_impressions(self, x):
        ih = [self.Impression(d) for d in x.split('#N#') if not d.startswith('#TAB#')]
        return ih

    def _load_data(self):
        logging.info('[+] loading data')
        self.data = []
        with open(self.config.training_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                self.data.append((
                    self._extract_impressions(line[2]) if line[2] else [],
                    self._extract_impressions(line[3]) if line[3] else []
                ))
        super(Seq2VecPaper, self)._load_data()
        logging.info('[-] loaded data')

    def train_gen(self):
        while True:
            for ih, _ in self.data:
                if ih:
                    ch = self.Window(self.docs, self.config.window_size)
                    if len(ih) > self.config.max_impression:
                        training_impressions = set(random.sample(ih, self.config.max_impression))

                        def get_trainable(x):
                            return x in training_impressions
                    else:
                        def get_trainable(_):
                            return True

                    for impression in ih:
                        trainable = get_trainable(impression)
                        for pos in impression.pos:
                            if ch.count and trainable:
                                clicked = ch.get_title()
                                yield clicked, self.docs[pos].title, 1
                                for neg in impression.negative_samples(self.config.negative_samples):
                                    yield clicked, self.docs[neg].title, 0
                            ch.push(pos)

    def valid_gen(self):
        while True:
            for ih1, ih2 in self.data:
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        for pos in impression.pos:
                            clicked = ch.get_title()
                            yield clicked, self.docs[pos].title, 1
                            for neg in impression.negative_samples(self.config.negative_samples):
                                yield clicked, self.docs[neg].title, 0
                        for pos in impression.pos:
                            ch.push(pos)

    def test_gen(self):
        def __gen__(_clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _clicked, doc.title, 0

        for ih1, ih2 in self.data:
            if ih1 and ih2:
                ch = self.Window(self.docs, self.config.window_size)
                for impression in ih1:
                    for pos in impression.pos:
                        ch.push(pos)
                for impression in ih2:
                    clicked = ch.get_title()
                    yield list(__gen__(clicked, impression))
                    for pos in impression.pos:
                        ch.push(pos)

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

                cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)
                emb = keras.layers.Dropout(self.config.dropout)(emb)
                c = cnn(emb)

                c = keras.layers.Lambda(
                    lambda x: x[0] * keras.backend.expand_dims(
                        keras.backend.switch(
                            keras.backend.equal(x[1], 0),
                            keras.backend.zeros_like(x[1], dtype=keras.backend.floatx()),
                            keras.backend.ones_like(x[1], dtype=keras.backend.floatx()))))([c, inp])

                cmask = keras.layers.Masking()(c)
                a = models.SimpleAttentionMaskSupport()(keras.layers.Dropout(self.config.dropout)(cmask))
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
        elif user_model == 'gru':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec)
        elif user_model == 'avg':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = models.GlobalAveragePoolingMaskSupport()(mask_clicked_vec)
        else:
            raise Exception('Unsupport user model')

        user_encoder = keras.Model(user_clicked_vec, user_vec, name='user_encoder')

        return user_encoder

    def score_encoder(self, user_vec, candidate_vec):
        join_vec = keras.layers.concatenate([user_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.user_embedding_dim, activation='relu', name='concat_dense')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid', name='socre_dense')(hidden)
        return logits

    def _build_model(self):
        inp_shape = self.config.title_shape

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

    def save_model(self):
        pass


class Seq2VecPaperDot(Seq2VecPaper):
    def score_encoder(self, user_vec, candidate_vec):
        score = keras.layers.Dot(([1, 1]))([user_vec, candidate_vec])
        logits = keras.layers.Activation('sigmoid')(score)
        return logits


class Seq2VecPaperId(Seq2VecPaper):
    def train_gen(self):
        while True:
            for user, (ih, _) in enumerate(self.data):
                if ih:
                    ch = self.Window(self.docs, self.config.window_size)
                    if len(ih) > self.config.max_impression:
                        training_impressions = set(random.sample(ih, self.config.max_impression))

                        def get_trainable(x):
                            return x in training_impressions
                    else:
                        def get_trainable(_):
                            return True

                    for impression in ih:
                        trainable = get_trainable(impression)
                        for pos in impression.pos:
                            if ch.count and trainable:
                                clicked = ch.get_title()
                                yield user, clicked, self.docs[pos].title, 1
                                for neg in impression.negative_samples(self.config.negative_samples):
                                    yield user, clicked, self.docs[neg].title, 0
                            ch.push(pos)

    def valid_gen(self):
        while True:
            for user, (ih1, ih2) in enumerate(self.data):
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        for pos in impression.pos:
                            clicked = ch.get_title()
                            yield user, clicked, self.docs[pos].title, 1
                            for neg in impression.negative_samples(self.config.negative_samples):
                                yield user, clicked, self.docs[neg].title, 0
                        for pos in impression.pos:
                            ch.push(pos)

    def test_gen(self):
        def __gen__(_user, _clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user, _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user, _clicked, doc.title, 0

        for user, (ih1, ih2) in enumerate(self.data):
            if ih1 and ih2:
                ch = self.Window(self.docs, self.config.window_size)
                for impression in ih1:
                    for pos in impression.pos:
                        ch.push(pos)
                for impression in ih2:
                    clicked = ch.get_title()
                    yield list(__gen__(user, clicked, impression))
                    for pos in impression.pos:
                        ch.push(pos)

    def get_user_encoder(self, window_size=None):
        if window_size is None:
            window_size = self.config.window_size
        user_idx = keras.Input((1,), name='user_idx')
        user_embedding_layer = keras.layers.Embedding(len(self.data), self.config.user_embedding_dim)

        user_clicked_vec = keras.Input((window_size, self.config.user_embedding_dim), name='user_clicked_vec')
        user_vec = keras.layers.Reshape((-1,))(user_embedding_layer(user_idx))

        user_model = self.config.arch

        if user_model == 'gru':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec)
            user_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'igru':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            user_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec, initial_state=user_vec)
        elif user_model == 'iigru':
            mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec, initial_state=user_vec)
            user_embedding_layer2 = keras.layers.Embedding(len(self.data), self.config.user_embedding_dim)
            user_vec2 = keras.layers.Reshape((-1,))(user_embedding_layer2(user_idx))
            user_vec = keras.layers.concatenate([clicked_vec, user_vec2])
        elif user_model == 'vo':
            pass
        else:
            raise Exception('Unsupport user model')

        user_encoder = keras.Model([user_idx, user_clicked_vec], user_vec, name='user_encoder')
        return user_encoder

    def _build_model(self):
        inp_shape = self.config.title_shape
        user_idx = keras.Input((1,))
        clicked = keras.Input((self.config.window_size, inp_shape))
        doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])
        user_encoder = self.get_user_encoder()
        user_vec = user_encoder([user_idx, clicked_vec])

        candidate = keras.Input((inp_shape,))
        candidate_vec = doc_encoder(candidate)

        logits = self.score_encoder(user_vec, candidate_vec)

        self.model = keras.Model([user_idx, clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )


class Seq2VecPaperSoftmax(Seq2VecPaper):
    def train_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for ih, _ in self.data:
                if ih:
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih:
                        for pos in impression.pos:
                            if ch.count:
                                clicked = ch.get_title()
                                yield [clicked, self.docs[pos].title] + [
                                    self.docs[neg].title for neg in
                                    impression.negative_samples(self.config.negative_samples)
                                ] + [label]
                            ch.push(pos)

    def valid_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for ih1, ih2 in self.data:
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        for pos in impression.pos:
                            clicked = ch.get_title()
                            yield [clicked, self.docs[pos].title] + [
                                self.docs[neg].title for neg in
                                impression.negative_samples(self.config.negative_samples)
                            ] + [label]
                        for pos in impression.pos:
                            ch.push(pos)

    def test_gen(self):
        def __gen__(_clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _clicked, doc.title, 0

        for ih1, ih2 in self.data:
            if ih1 and ih2:
                ch = self.Window(self.docs, self.config.window_size)
                for impression in ih1:
                    for pos in impression.pos:
                        ch.push(pos)
                for impression in ih2:
                    clicked = ch.get_title()
                    yield list(__gen__(clicked, impression))
                    for pos in impression.pos:
                        ch.push(pos)

    def _score_model(self, u, d):
        u_input = keras.Input((u.shape[-1].value,))
        d_input = keras.Input((d.shape[-1].value,))
        if self.config.score_model == 'dot':
            score = keras.layers.dot([u_input, d_input], -1)
        elif self.config.score_model == 'dnn':
            hid = keras.layers.concatenate([u_input, d_input])
            hid = keras.layers.Dense(self.config.user_embedding_dim, activation='relu')(hid)
            score = keras.layers.Dense(1)(hid)
        elif self.config.score_model == 'ddot':
            u_hid = keras.layers.Dense(self.config.user_embedding_dim, activation='tanh')(u_input)
            d_hid = keras.layers.Dense(self.config.user_embedding_dim, activation='tanh')(d_input)
            score = keras.layers.dot([u_hid, d_hid], -1)
        else:
            raise NotImplementedError
        self.score_model = keras.Model([u_input, d_input], score)

    def score_encoder(self, user_vec, candidate_vecs):
        self._score_model(user_vec, candidate_vecs[0])
        logits = [self.score_model([user_vec, candidate_vec]) for candidate_vec in candidate_vecs]
        logits = keras.layers.Activation(keras.activations.softmax, name='ranking')(keras.layers.concatenate(logits))
        return logits

    def _build_model(self):
        inp_shape = self.config.title_shape

        clicked = keras.Input((self.config.window_size, inp_shape))
        doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])
        user_encoder = self.get_user_encoder()
        user_vec = user_encoder(clicked_vec)

        candidates = [keras.Input((inp_shape,)) for _ in range(self.config.negative_samples + 1)]
        candidate_vecs = [doc_encoder(candidate) for candidate in candidates]

        logits = self.score_encoder(user_vec, candidate_vecs)

        self.model = keras.Model([clicked] + candidates, logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy]
        )

        candidate_one = keras.Input((inp_shape,))
        candidate_one_vec = doc_encoder(candidate_one)
        score = keras.layers.Activation(keras.activations.sigmoid)(
            self.score_model([user_vec, candidate_one_vec]))
        self.test_model = keras.Model([clicked, candidate_one], score)

    def callback(self, epoch):
        keras.backend.set_value(self.model.optimizer.lr,
                                keras.backend.get_value(self.model.optimizer.lr) * self.config.learning_rate_decay)

        if epoch or True:
            self.model, self.test_model = self.test_model, self.model

            def __gen__(x):
                for i, (y_pred, y_true) in zip(range(x), self.test):
                    auc = roc_auc_score(y_true, y_pred)
                    ndcgx = utils.ndcg_score(y_true, y_pred, 10)
                    ndcgv = utils.ndcg_score(y_true, y_pred, 5)
                    mrr = utils.mrr_score(y_true, y_pred)
                    pos = np.sum(y_true)
                    size = len(y_true)
                    yield auc, ndcgx, ndcgv, mrr, pos, size, i

            values = [np.mean(x) for x in zip(*__gen__(self.config.validation_impression))]
            utils.logging_evaluation(dict(auc=values[0], ndcgx=values[1], ndcgv=values[2], mrr=values[3]))
            utils.logging_evaluation(dict(pos=values[4], size=values[5], num=values[6] * 2 + 1))

            if epoch == self.config.epochs - 1:
                self.is_training = False
                values = [np.mean(x) for x in zip(*__gen__(self.config.testing_impression))]
                utils.logging_evaluation(dict(auc=values[0], ndcgx=values[1], ndcgv=values[2], mrr=values[3]))
                utils.logging_evaluation(dict(pos=values[4], size=values[5], num=values[6] * 2 + 1))

            self.model, self.test_model = self.test_model, self.model


class Seq2VecPaperSoftmaxId(Seq2VecPaperSoftmax):
    def train_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih, _) in enumerate(self.data):
                if ih:
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih:
                        for pos in impression.pos:
                            if ch.count:
                                clicked = ch.get_title()
                                yield [user, clicked, self.docs[pos].title] + [
                                    self.docs[neg].title for neg in
                                    impression.negative_samples(self.config.negative_samples)
                                ] + [label]
                            ch.push(pos)

    def valid_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih1, ih2) in enumerate(self.data):
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        for pos in impression.pos:
                            clicked = ch.get_title()
                            yield [user, clicked, self.docs[pos].title] + [
                                self.docs[neg].title for neg in
                                impression.negative_samples(self.config.negative_samples)
                            ] + [label]
                        for pos in impression.pos:
                            ch.push(pos)

    def test_gen(self):
        def __gen__(_user, _clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user, _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user, _clicked, doc.title, 0

        for user, (ih1, ih2) in enumerate(self.data):
            if ih1 and ih2:
                ch = self.Window(self.docs, self.config.window_size)
                for impression in ih1:
                    for pos in impression.pos:
                        ch.push(pos)
                for impression in ih2:
                    clicked = ch.get_title()
                    yield list(__gen__(user, clicked, impression))
                    for pos in impression.pos:
                        ch.push(pos)

    def get_user_encoder(self, window_size=None):
        if window_size is None:
            window_size = self.config.window_size

        user_idx = keras.Input((1,), name='user_idx')
        user_embedding_layer = keras.layers.Embedding(len(self.data), self.config.user_embedding_dim)
        user_clicked_vec = keras.Input((window_size, self.config.user_embedding_dim), name='user_clicked_vec')
        user_vec = keras.layers.Reshape((self.config.user_embedding_dim,))(user_embedding_layer(user_idx))
        mask_clicked_vec = keras.layers.Masking()(user_clicked_vec)

        user_model = self.config.arch

        if user_model == 'gru':
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec)
            user_vec = keras.layers.concatenate([clicked_vec, user_vec])
            user_vec = keras.layers.Dense(self.config.user_embedding_dim)(user_vec)
        elif user_model == 'ngru':
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec)
            user_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'hgru':
            user_embedding_layer = keras.layers.Embedding(len(self.data), self.config.user_embedding_dim // 2)
            user_vec = keras.layers.Reshape((self.config.user_embedding_dim // 2,))(user_embedding_layer(user_idx))
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim // 2)(mask_clicked_vec)
            user_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'dgru':
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec)
            user_vec = keras.layers.Dropout(0.5, noise_shape=(None, 1))(user_vec)
            user_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'igru':
            user_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec, initial_state=user_vec)
        elif user_model == 'iigru':
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec, initial_state=user_vec)
            user_embedding_layer2 = keras.layers.Embedding(len(self.data), self.config.user_embedding_dim)
            user_vec2 = keras.layers.Reshape((self.config.user_embedding_dim,))(user_embedding_layer2(user_idx))
            user_vec = keras.layers.concatenate([clicked_vec, user_vec2])
            user_vec = keras.layers.Dense(self.config.user_embedding_dim)(user_vec)
        elif user_model == 'vo':
            pass
        elif user_model == 'pgru':
            clicked_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec)
            user_vec = keras.layers.add([clicked_vec, user_vec])
        elif user_model == 'nigru':
            user_vec = keras.layers.GRU(self.config.user_embedding_dim)(mask_clicked_vec)
        elif user_model == 'niavg':
            user_vec = models.GlobalAveragePoolingMaskSupport()(mask_clicked_vec)
        else:
            raise Exception('Unsupport user model')

        user_encoder = keras.Model([user_idx, user_clicked_vec], user_vec, name='user_encoder')
        return user_encoder

    def _build_model(self):
        inp_shape = self.config.title_shape

        user = keras.Input((1,))
        clicked = keras.Input((self.config.window_size, inp_shape))
        self.doc_encoder = doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])
        self.user_encoder = user_encoder = self.get_user_encoder()
        user_vec = user_encoder([user, clicked_vec])

        candidates = [keras.Input((inp_shape,)) for _ in range(self.config.negative_samples + 1)]
        candidate_vecs = [doc_encoder(candidate) for candidate in candidates]

        logits = self.score_encoder(user_vec, candidate_vecs)

        self.model = keras.Model([user, clicked] + candidates, logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy]
        )

        candidate_one = keras.Input((inp_shape,))
        candidate_one_vec = doc_encoder(candidate_one)
        score = keras.layers.Activation(keras.activations.sigmoid)(
            self.score_model([user_vec, candidate_one_vec]))
        self.test_model = keras.Model([user, clicked, candidate_one], score)


class Seq2VecPaperSoftmaxDays(Seq2VecPaperSoftmax):
    class Window:
        __slots__ = ['docs', 'click_history', 'click_history_time', 'delta']
        zero = datetime.strptime('01/01/2000', '%m/%d/%Y')

        def __init__(self, docs, window_size, delta):
            self.docs = docs
            self.delta = timedelta(days=delta)
            self.click_history = [0 for _ in range(window_size)]
            self.click_history_time = [self.zero for _ in range(window_size)]

        def get_title(self, time):
            return np.stack([
                self.docs[i if t >= time else 0].title for i, t in zip(self.click_history, self.click_history_time)])

        def push(self, doc, time):
            self.click_history.append(doc)
            self.click_history.pop(0)
            self.click_history_time.append(time + self.delta)
            self.click_history_time.pop(0)

        @property
        def window_size(self):
            return len(self.click_history)

        def count(self, time):
            return sum(1 for t in self.click_history_time if t >= time)

    def train_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for ih, _ in self.data:
                if ih:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih:
                        for pos in impression.pos:
                            if ch.count(impression.time):
                                clicked = ch.get_title(impression.time)
                                yield [clicked, self.docs[pos].title] + [
                                    self.docs[neg].title for neg in
                                    impression.negative_samples(self.config.negative_samples)
                                ] + [label]
                            ch.push(pos, impression.time)

    def valid_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for ih1, ih2 in self.data:
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos, impression.time)
                    for impression in ih2:
                        for pos in impression.pos:
                            clicked = ch.get_title(impression.time)
                            yield [clicked, self.docs[pos].title] + [
                                self.docs[neg].title for neg in
                                impression.negative_samples(self.config.negative_samples)
                            ] + [label]
                        for pos in impression.pos:
                            ch.push(pos, impression.time)

    def test_gen(self):
        def __gen__(_clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _clicked, doc.title, 0

        for ih1, ih2 in self.data:
            if ih1 and ih2:
                ch = self.Window(self.docs, self.config.window_size, self.config.days)
                for impression in ih1:
                    for pos in impression.pos:
                        ch.push(pos, impression.time)
                for impression in ih2:
                    clicked = ch.get_title(impression.time)
                    yield list(__gen__(clicked, impression))
                    for pos in impression.pos:
                        ch.push(pos, impression.time)


class Seq2VecPaperSoftmaxDaysId(Seq2VecPaperSoftmaxId):
    class News:
        __slots__ = ['title', 'body', 'vertical', 'subvertical']

        def __init__(self, title, body, vertical, subvertical):
            self.title = title
            self.body = body
            self.vertical = vertical
            self.subvertical = subvertical

    class Window:
        __slots__ = ['docs', 'click_history', 'click_history_time', 'delta']
        zero = datetime.strptime('01/01/2000', '%m/%d/%Y')

        def __init__(self, docs, window_size, delta):
            self.docs = docs
            self.delta = timedelta(days=delta)
            self.click_history = [0 for _ in range(window_size)]
            self.click_history_time = [self.zero for _ in range(window_size)]

        def get_title(self, time):
            return np.stack([
                self.docs[i if t >= time else 0].title for i, t in zip(self.click_history, self.click_history_time)])

        def get_vertical(self, time):
            return [self.docs[i if t >= time else 0].vertical
                    for i, t in zip(self.click_history, self.click_history_time)]

        def push(self, doc, time):
            self.click_history.append(doc)
            self.click_history.pop(0)
            self.click_history_time.append(time + self.delta)
            self.click_history_time.pop(0)

        @property
        def window_size(self):
            return len(self.click_history)

        def count(self, time):
            return sum(1 for t in self.click_history_time if t >= time)

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

        with utils.open(self.config.doc_meta_input) as file:
            docs = [line.strip('\n').split('\t') for line in file]

        self.docs = {
            int(line[1]): self.News(
                title_parser(line[4])[0],
                body_parser(line[5])[0],
                utils.get_vertical(line[2]),
                utils.get_subvertical(line[3])
            ) for line in docs}

        self.doc_count = max(self.docs.keys()) + 1
        doc_example = self.docs[self.doc_count - 1]
        self.docs[0] = self.News(
            np.zeros_like(doc_example.title),
            np.zeros_like(doc_example.body),
            utils.get_vertical('N/A'),
            utils.get_subvertical('N/A')
        )

        logging.info("[-] loaded docs metadata")

    def train_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih, _) in enumerate(self.data):
                if ih:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih:
                        for pos in impression.pos:
                            if ch.count(impression.time):
                                clicked = ch.get_title(impression.time)
                                yield [user, clicked, self.docs[pos].title] + [
                                    self.docs[neg].title for neg in
                                    impression.negative_samples(self.config.negative_samples)
                                ] + [label]
                            ch.push(pos, impression.time)

    def valid_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih1, ih2) in enumerate(self.data):
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos, impression.time)
                    for impression in ih2:
                        for pos in impression.pos:
                            clicked = ch.get_title(impression.time)
                            yield [user, clicked, self.docs[pos].title] + [
                                self.docs[neg].title for neg in
                                impression.negative_samples(self.config.negative_samples)
                            ] + [label]
                        for pos in impression.pos:
                            ch.push(pos, impression.time)

    def test_gen(self):
        def __gen__(_user, _clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user, _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user, _clicked, doc.title, 0

        for user, (ih1, ih2) in enumerate(self.data):
            if ih1 and ih2:
                ch = self.Window(self.docs, self.config.window_size, self.config.days)
                for impression in ih1:
                    for pos in impression.pos:
                        ch.push(pos, impression.time)
                for impression in ih2:
                    clicked = ch.get_title(impression.time)
                    yield list(__gen__(user, clicked, impression))
                    for pos in impression.pos:
                        ch.push(pos, impression.time)


class Seq2VecPaperSoftmaxDaysIdVertSup(Seq2VecPaperSoftmaxDaysId):
    def train_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih, _) in enumerate(self.data):
                if ih:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih:
                        for pos in impression.pos:
                            if ch.count(impression.time):
                                clicked = ch.get_title(impression.time)
                                clicked_vert = ch.get_vertical(impression.time)
                                negs = impression.negative_samples(self.config.negative_samples)
                                yield [user, clicked, self.docs[pos].title] + [
                                    self.docs[neg].title for neg in negs
                                ] + [label, keras.utils.to_categorical(
                                    clicked_vert
                                    + [self.docs[pos].vertical] + [self.docs[neg].vertical for neg in negs],
                                    num_classes=len(utils.verticals))]
                            ch.push(pos, impression.time)

    def valid_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih1, ih2) in enumerate(self.data):
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos, impression.time)
                    for impression in ih2:
                        for pos in impression.pos:
                            clicked = ch.get_title(impression.time)
                            clicked_vert = ch.get_vertical(impression.time)
                            negs = impression.negative_samples(self.config.negative_samples)
                            yield [user, clicked, self.docs[pos].title] + [
                                self.docs[neg].title for neg in negs
                            ] + [label, keras.utils.to_categorical(
                                clicked_vert
                                + [self.docs[pos].vertical] + [self.docs[neg].vertical for neg in negs],
                                num_classes=len(utils.verticals))]
                        for pos in impression.pos:
                            ch.push(pos, impression.time)

    @property
    def train(self):
        pool = []
        size = self.config.batch_size * 100
        gen = self.train_gen()
        while True:
            pool.append(next(gen))
            if len(pool) >= size:
                np.random.shuffle(pool)
                batch = [np.stack(x) for x in zip(*pool[:self.config.batch_size])]
                yield batch[:-2], batch[-2:]
                pool = pool[self.config.batch_size:]

    @property
    def valid(self):
        gen = self.valid_gen()
        while True:
            batch = [np.stack(x) for x in zip(*(next(gen) for _ in range(self.config.batch_size)))]
            yield batch[:-2], batch[-2:]

    def get_vertical_classifier(self, input_shape):
        inp = keras.Input((input_shape,))
        hid = keras.layers.Dense(self.config.hidden_dim, activation='relu')(inp)
        logits = keras.layers.Dense(len(utils.verticals), activation='softmax')(hid)
        return keras.Model(inp, logits)

    def _build_model(self):
        inp_shape = self.config.title_shape

        user = keras.Input((1,))
        clicked = keras.Input((self.config.window_size, inp_shape))
        doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])
        user_encoder = self.get_user_encoder()
        user_vec = user_encoder([user, clicked_vec])

        candidates = [keras.Input((inp_shape,)) for _ in range(self.config.negative_samples + 1)]
        candidate_vecs = [doc_encoder(candidate) for candidate in candidates]

        logits = self.score_encoder(user_vec, candidate_vecs)

        vertical_classifier = keras.layers.TimeDistributed(
            self.get_vertical_classifier(clicked_vec.get_shape()[-1].value), name='vert')

        expand_vec = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=-2))
        candidate_vecs = keras.layers.concatenate([expand_vec(x) for x in candidate_vecs], axis=-2)
        # mask_candidates = models.ComputeMasking()(candidate_vecs)

        doc_vecs = keras.layers.concatenate([clicked_vec, candidate_vecs], axis=-2)
        # doc_vecs = clicked_vec
        verticals = vertical_classifier(doc_vecs)
        # verticals = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(keras.backend.concatenate(
        #     [x[1], x[2]])))(
        #     [verticals, mask, mask_candidates])
        # verticals = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]), name='vert')(
        #     [verticals, mask])

        self.model = keras.Model([user, clicked] + candidates, [logits, verticals])
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=[keras.losses.categorical_crossentropy, keras.losses.categorical_crossentropy],
            loss_weights=[1, self.config.gain],
            metrics=[keras.metrics.categorical_accuracy])

        candidate_one = keras.Input((inp_shape,))
        candidate_one_vec = doc_encoder(candidate_one)
        score = keras.layers.Activation(keras.activations.sigmoid)(
            self.score_model([user_vec, candidate_one_vec]))
        self.test_model = keras.Model([user, clicked, candidate_one], score)


class Seq2VecPaperSoftmaxDaysIdVertAlt(Seq2VecPaperSoftmaxDaysId):
    def __init__(self, config: settings.Config):
        super(Seq2VecPaperSoftmaxDaysIdVertAlt, self).__init__(config)
        self.round = self.config.round
        self.config.epochs *= self.round

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
        self.docs = {int(line[1]): self.News(
            title_parser(line[4])[0],
            body_parser(line[5])[0],
            line[2],
            line[3],
        ) for line in lines}
        self.verticals = list(set(news.vertical for news in self.docs.values()))
        self.subverticals = list(set(news.subvertical for news in self.docs.values()))
        self.data_verticals = keras.utils.to_categorical(
            np.array([self.verticals.index(news.vertical) for news in self.docs.values()]))
        self.data_titles = np.stack([news.title for news in self.docs.values()])

        data = np.arange(len(self.docs))
        np.random.shuffle(data)
        self.train_index = data[:len(self.docs) // 10]
        self.valid_index = data[len(self.docs) // 10:]

        self.doc_count = max(self.docs.keys()) + 1
        doc_example = self.docs[self.doc_count - 1]
        self.docs[0] = self.News(
            np.zeros_like(doc_example.title),
            np.zeros_like(doc_example.body),
            'N/A', 'N/A', )
        logging.info("[-] loaded docs metadata")

    @property
    def training_step(self):
        if self.train_seq:
            return self.config.training_step
        else:
            return len(self.train_index) // self.config.batch_size

    @property
    def validation_step(self):
        if self.train_seq:
            return self.config.validation_step
        else:
            return len(self.valid_index) // self.config.batch_size

    @training_step.setter
    def training_step(self, value):
        pass

    @validation_step.setter
    def validation_step(self, value):
        pass

    @property
    def train_vert(self):
        while True:
            np.random.shuffle(self.train_index)
            data_titles = self.data_titles[self.train_index]
            data_verticals = self.data_verticals[self.train_index]
            for batch in range(self.config.batch_size, len(self.train_index), self.config.batch_size):
                yield data_titles[batch - self.config.batch_size:batch], \
                      data_verticals[batch - self.config.batch_size:batch]

    @property
    def valid_vert(self):
        while True:
            data_titles = self.data_titles[self.valid_index]
            data_verticals = self.data_verticals[self.valid_index]
            for batch in range(self.config.batch_size, len(self.valid_index), self.config.batch_size):
                yield data_titles[batch - self.config.batch_size:batch], \
                      data_verticals[batch - self.config.batch_size:batch]

    @property
    def train(self):
        train_vert = self.train_vert
        train = super(Seq2VecPaperSoftmaxDaysIdVertAlt, self).train
        while True:
            if self.train_seq:
                yield next(train)
            else:
                yield next(train_vert)

    @property
    def valid(self):
        valid_vert = self.valid_vert
        valid = super(Seq2VecPaperSoftmaxDaysIdVertAlt, self).valid
        if self.train_seq:
            return valid
        else:
            return valid_vert

    def callback(self, epoch):
        if self.train_seq:
            super(Seq2VecPaperSoftmaxDaysIdVertAlt, self).callback(epoch)
        else:
            if epoch % self.round == self.round - 2:
                keras.backend.set_value(
                    self.model.optimizer.lr,
                    keras.backend.get_value(self.model.optimizer.lr) * self.config.learning_rate_decay)

    def callback_valid(self, epoch):
        self.train_seq = epoch % self.round == self.round - 2
        if self.train_seq:
            self.model = self.seq_model
        else:
            self.model = self.vert_model

    def _build_model(self):
        self.train_seq = False

        super(Seq2VecPaperSoftmaxDaysIdVertAlt, self)._build_model()
        self.seq_model = self.model
        title = keras.Input((self.config.title_shape,))
        hidden = self.doc_encoder(title)
        logits = keras.layers.Dense(len(self.verticals), activation='softmax')(hidden)

        self.model = self.vert_model = keras.Model(title, logits)
        self.vert_model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy])


class Seq2VecPaperSoftmaxDaysIdVert(Seq2VecPaperSoftmaxDaysId):
    def train_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih, _) in enumerate(self.data):
                if ih:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih:
                        for pos in impression.pos:
                            if ch.count(impression.time):
                                negs = impression.negative_samples(self.config.negative_samples)
                                yield [user,
                                       ch.get_title(impression.time),
                                       ch.get_vertical(impression.time),
                                       self.docs[pos].title] + \
                                      [self.docs[neg].title for neg in negs] + \
                                      [self.docs[pos].vertical] + \
                                      [self.docs[neg].vertical for neg in negs] + \
                                      [label]
                            ch.push(pos, impression.time)

    def valid_gen(self):
        label = [1] + [0 for _ in range(self.config.negative_samples)]
        while True:
            for user, (ih1, ih2) in enumerate(self.data):
                if ih1 and ih2:
                    ch = self.Window(self.docs, self.config.window_size, self.config.days)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos, impression.time)
                    for impression in ih2:
                        for pos in impression.pos:
                            negs = impression.negative_samples(self.config.negative_samples)
                            yield [user,
                                   ch.get_title(impression.time),
                                   ch.get_vertical(impression.time),
                                   self.docs[pos].title] + \
                                  [self.docs[neg].title for neg in negs] + \
                                  [self.docs[pos].vertical] + \
                                  [self.docs[neg].vertical for neg in negs] + \
                                  [label]
                        for pos in impression.pos:
                            ch.push(pos, impression.time)

    def test_gen(self):
        def __gen__(_user, _clicked_title, _clicked_vert, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user, _clicked_title, _clicked_vert, doc.title, doc.vertical, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user, _clicked_title, _clicked_vert, doc.title, doc.vertical, 0

        for user, (ih1, ih2) in enumerate(self.data):
            if ih1 and ih2:
                ch = self.Window(self.docs, self.config.window_size, self.config.days)
                for impression in ih1:
                    for pos in impression.pos:
                        ch.push(pos, impression.time)
                for impression in ih2:
                    clicked = ch.get_title(impression.time)
                    clicked_vert = ch.get_vertical(impression.time)
                    yield list(__gen__(user, clicked, clicked_vert, impression))
                    for pos in impression.pos:
                        ch.push(pos, impression.time)

    def get_user_encoder(self, window_size=None):
        self.config.user_embedding_dim += self.config.vertical_embedding_dim
        user_encoder = super(Seq2VecPaperSoftmaxDaysIdVert, self).get_user_encoder(window_size)
        self.config.user_embedding_dim -= self.config.vertical_embedding_dim
        return user_encoder

    def _build_model(self):
        inp_shape = self.config.title_shape

        user = keras.Input((1,))
        clicked = keras.Input((self.config.window_size, inp_shape))
        clicked_vert = keras.Input((self.config.window_size,))

        vert_embedding_layer = keras.layers.Embedding(len(utils.verticals), self.config.vertical_embedding_dim)
        vert_encoder = keras.Sequential(
            [vert_embedding_layer, keras.layers.Reshape((self.config.vertical_embedding_dim,))])
        vert_encoder_dist = keras.layers.TimeDistributed(vert_encoder)

        self.doc_encoder = doc_encoder = self.get_doc_encoder()
        user_encoder_dist = keras.layers.TimeDistributed(doc_encoder)
        clicked_vec_raw = user_encoder_dist(clicked)
        clicked_vert_raw = vert_encoder_dist(keras.layers.Lambda(keras.backend.expand_dims)(clicked_vert))
        clicked_vec_raw = keras.layers.concatenate([clicked_vec_raw, clicked_vert_raw])

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec_raw, mask])
        self.user_encoder = user_encoder = self.get_user_encoder()
        user_vec = user_encoder([user, clicked_vec])

        candidates = [keras.Input((inp_shape,)) for _ in range(self.config.negative_samples + 1)]
        candidate_verts = [keras.Input((1,)) for _ in range(self.config.negative_samples + 1)]
        candidate_vecs = [doc_encoder(candidate) for candidate in candidates]
        candidate_vert_vecs = [vert_encoder(candidate_vert) for candidate_vert in candidate_verts]
        candidate_vecs = [keras.layers.concatenate([a, b]) for a, b in zip(candidate_vecs, candidate_vert_vecs)]

        logits = self.score_encoder(user_vec, candidate_vecs)

        self.model = keras.Model([user, clicked, clicked_vert] + candidates + candidate_verts, logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy]
        )

        candidate_one = keras.Input((inp_shape,))
        candidate_one_vert = keras.Input((1,))
        candidate_one_vec = doc_encoder(candidate_one)
        candidate_one_vert_vec = vert_encoder(candidate_one_vert)
        candidate_one_vec = keras.layers.concatenate([candidate_one_vec, candidate_one_vert_vec])
        score = keras.layers.Activation(keras.activations.sigmoid)(
            self.score_model([user_vec, candidate_one_vec]))
        self.test_model = keras.Model([user, clicked, clicked_vert, candidate_one, candidate_one_vert], score)
