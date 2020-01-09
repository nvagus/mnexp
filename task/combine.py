from .seq2vec import *


class Seq2VecCombineEncoder(Seq2VecForward):
    def _build_model(self):
        self.doc_encoder = doc_encoder = self.get_doc_encoder()
        user_encoder = keras.layers.TimeDistributed(doc_encoder, name='long')
        click_encoder = keras.layers.TimeDistributed(doc_encoder, name='short')

        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        candidate = keras.Input((self.config.title_shape,))

        long = keras.layers.Lambda(lambda x: x[:, :-20, :])(clicked)
        short = keras.layers.Lambda(lambda x: x[:, -20:, :])(clicked)

        user_vec = user_encoder(long)
        clicked_vec = click_encoder(short)
        candidate_vec = doc_encoder(candidate)

        clicked_mask = models.ComputeMasking(0)(short)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec, clicked_mask])

        user_mask = models.ComputeMasking(0)(long)
        user_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([user_vec, user_mask])

        user_vec = models.GlobalAveragePoolingMasked(user_mask)(user_vec)

        user_model = self.config.arch
        logging.info('[!] Selecting User Model: {}'.format(user_model))
        if user_model == 'gru':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.GRU(self.config.hidden_dim)(clicked_vec)
            clicked_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'lstm':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.LSTM(self.config.hidden_dim)(clicked_vec)
            clicked_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'igru':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.GRU(self.config.hidden_dim)(clicked_vec, initial_state=user_vec)
        elif user_model == 'ilstm':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.LSTM(self.config.hidden_dim)(
                clicked_vec, initial_state=[self.config.hidden_dim,
                                            keras.layers.Lambda(keras.backend.zeros_like)(self.config.hidden_dim)])
        else:
            if user_model != 'avg':
                logging.warning('[!] arch {} not found, using average by default'.format(user_model))
            clicked_vec = models.GlobalAveragePoolingMasked(clicked_mask)(clicked_vec)

        join_vec = keras.layers.concatenate([clicked_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.hidden_dim, activation='relu')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid')(hidden)

        self.model = keras.Model([clicked, candidate], logits)
        if self.__class__ == Seq2VecCombineEncoder:
            self.model.compile(
                optimizer=keras.optimizers.Adam(self.config.learning_rate),
                loss=self.loss,
                metrics=[utils.auc_roc]
            )
        else:
            return self.model


class Seq2VecCombine(Seq2VecForward):
    def _load_users(self):
        logging.info('[+] loading users')
        with utils.open(self.config.result_input) as file:
            lines = [line.strip('\n').split('\t') for line in file]
        self.users = {
            line[0] + line[1]: np.array(line[2].split(), np.float32)
            for line in lines
        }
        self.user_vec_dim = len(self.users[next(iter(self.users.keys()))])
        logging.info('[-] loaded {} users'.format(len(self.users)))

    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    user_vec = self.users[line[0] + line[1]]
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield user_vec, clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user_vec, clicked, self.docs[neg].title, 0
                                ch.push(pos)

    def valid_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for line in file:
                    line = line.strip('\n').split('\t')
                    user_vec = self.users[line[0] + line[1]]
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
                                    yield user_vec, clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user_vec, clicked, self.docs[neg].title, 0
                                ch.push(pos)

    def test_gen(self):
        def __gen__(_user_vec, _clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user_vec, _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user_vec, _clicked, doc.title, 0

        with open(self.config.training_data_input if self.is_training else self.config.testing_data_input) as file:
            for line in file:
                line = line.strip('\n').split('\t')
                user_vec = self.users[line[0] + line[1]]
                if line[2] and line[3]:
                    ih1 = self._extract_impressions(line[2])
                    ih2 = self._extract_impressions(line[3])
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        clicked = ch.get_title()
                        yield list(__gen__(user_vec, clicked, impression))
                        for pos in impression.pos:
                            ch.push(pos)

    def _build_model(self):
        self.doc_encoder = doc_encoder = self.get_doc_encoder()
        user_encoder = keras.layers.TimeDistributed(doc_encoder)

        user_vec = keras.Input((self.user_vec_dim,))
        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        candidate = keras.Input((self.config.title_shape,))

        clicked_vec = user_encoder(clicked)
        candidate_vec = doc_encoder(candidate)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec, mask])

        user_model = self.config.arch
        logging.info('[!] Selecting User Model: {}'.format(user_model))
        if user_model == 'gru':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.GRU(self.user_vec_dim)(clicked_vec)
            clicked_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'lstm':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.LSTM(self.user_vec_dim)(clicked_vec)
            clicked_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'igru':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.GRU(self.user_vec_dim)(clicked_vec, initial_state=user_vec)
        elif user_model == 'ilstm':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.LSTM(self.user_vec_dim)(
                clicked_vec,
                initial_state=[user_vec, keras.layers.Lambda(keras.backend.zeros_like)(user_vec)])
        else:
            if user_model != 'avg':
                logging.warning('[!] arch {} not found, using average by default'.format(user_model))
            clicked_vec = models.GlobalAveragePoolingMasked(mask)(clicked_vec)

        join_vec = keras.layers.concatenate([clicked_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.hidden_dim, activation='relu')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid')(hidden)

        self.model = keras.Model([user_vec, clicked, candidate], logits)
        if self.__class__ == Seq2VecCombine:
            self.model.compile(
                optimizer=keras.optimizers.Adam(self.config.learning_rate),
                loss=self.loss,
                metrics=[utils.auc_roc]
            )
        else:
            return self.model


class Seq2VecCombineId(Seq2VecForward):
    def _load_users(self):
        logging.info('[+] loading users')
        with open(self.config.training_data_input) as file:
            i = 0
            for i, _ in enumerate(file):
                pass
        self.user_count = i
        logging.info('[-] loaded {} users'.format(self.user_count))

    def train_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for user, line in enumerate(file):
                    line = line.strip('\n').split('\t')
                    if line[2]:
                        ih = self._extract_impressions(line[2])
                        ch = self.Window(self.docs, self.config.window_size)
                        for impression in ih:
                            for pos in impression.pos:
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield user, clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user, clicked, self.docs[neg].title, 0
                                ch.push(pos)

    def valid_gen(self):
        while True:
            with open(self.config.training_data_input) as file:
                for user, line in enumerate(file):
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
                                    clicked = ch.get_title()
                                    yield user, clicked, self.docs[pos].title, 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield user, clicked, self.docs[neg].title, 0
                                ch.push(pos)

    def test_gen(self):
        def __gen__(_user_vec, _clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _user_vec, _clicked, doc.title, 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _user_vec, _clicked, doc.title, 0

        with open(self.config.training_data_input if self.is_training else self.config.testing_data_input) as file:
            for user, line in enumerate(file):
                line = line.strip('\n').split('\t')
                if line[2] and line[3]:
                    ih1 = self._extract_impressions(line[2])
                    ih2 = self._extract_impressions(line[3])
                    ch = self.Window(self.docs, self.config.window_size)
                    for impression in ih1:
                        for pos in impression.pos:
                            ch.push(pos)
                    for impression in ih2:
                        clicked = ch.get_title()
                        yield list(__gen__(user, clicked, impression))
                        for pos in impression.pos:
                            ch.push(pos)

    def _build_model(self):
        self.doc_encoder = doc_encoder = self.get_doc_encoder()
        self.user_embedding = keras.layers.Embedding(self.user_count, self.config.user_embedding_dim)
        user_encoder = keras.layers.TimeDistributed(doc_encoder)

        user_idx = keras.Input((1,))
        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        candidate = keras.Input((self.config.title_shape,))

        user_vec = keras.layers.Reshape((-1,))(self.user_embedding(user_idx))
        clicked_vec = user_encoder(clicked)
        candidate_vec = doc_encoder(candidate)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec, mask])

        user_model = self.config.arch
        logging.info('[!] Selecting User Model: {}'.format(user_model))
        if user_model == 'gru':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.GRU(self.config.hidden_dim)(clicked_vec)
            clicked_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'lstm':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.LSTM(self.config.hidden_dim)(clicked_vec)
            clicked_vec = keras.layers.concatenate([clicked_vec, user_vec])
        elif user_model == 'igru':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.GRU(self.config.hidden_dim)(clicked_vec, initial_state=user_vec)
        elif user_model == 'ilstm':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = keras.layers.LSTM(self.config.hidden_dim)(
                clicked_vec,
                initial_state=[user_vec, keras.layers.Lambda(keras.backend.zeros_like)(user_vec)])
        elif user_model == 'vo':
            clicked_vec = user_vec
        else:
            if user_model != 'avg':
                logging.warning('[!] arch {} not found, using average by default'.format(user_model))
            clicked_vec = models.GlobalAveragePoolingMasked(mask)(clicked_vec)

        join_vec = keras.layers.concatenate([clicked_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.hidden_dim, activation='relu')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid')(hidden)

        self.model = keras.Model([user_idx, clicked, candidate], logits)
        if self.__class__ == Seq2VecCombineId:
            self.model.compile(
                optimizer=keras.optimizers.Adam(self.config.learning_rate),
                loss=self.loss,
                metrics=[utils.auc_roc]
            )
        else:
            return self.model
