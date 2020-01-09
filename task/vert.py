from .seq2vec import *


class Seq2VecVert(Seq2VecForward):
    class News:
        __slots__ = ['vertical', 'sub_vertical', 'title', 'body']

        def __init__(self, vertical, sub_vertical, title, body):
            self.vertical = vertical
            self.sub_vertical = sub_vertical
            self.title = title
            self.body = body

    def __init__(self, config: settings.Config):
        super(Seq2VecVert, self).__init__(config)
        self.round = 6
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
            line[2],
            line[3],
            title_parser(line[4])[0],
            body_parser(line[5])[0],
        ) for line in lines}
        self.verticals = list(set(news.vertical for news in self.docs.values()))
        self.sub_verticals = list(set(news.sub_vertical for news in self.docs.values()))
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
            'vertical', 'sub_vertical',
            np.zeros_like(doc_example.title),
            np.zeros_like(doc_example.body))
        logging.info("[-] loaded docs metadata")

    def _load_data(self):
        pass

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
        train = super(Seq2VecVert, self).train
        while True:
            if self.train_seq:
                yield next(train)
            else:
                yield next(train_vert)

    @property
    def valid(self):
        valid_vert = self.valid_vert
        valid = super(Seq2VecVert, self).valid
        if self.train_seq:
            return valid
        else:
            return valid_vert

    def _build_model(self):
        self.train_seq = False

        self.seq_model = super(Seq2VecVert, self)._build_model()

        title = keras.Input((self.config.title_shape,))
        hidden = self.doc_encoder(title)
        logits = keras.layers.Dense(len(self.verticals), activation='softmax')(hidden)

        self.model = self.vert_model = keras.Model(title, logits)
        self.seq_model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )
        self.vert_model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy])

    def callback(self, epoch):
        if self.train_seq:
            super(Seq2VecVert, self).callback(epoch // self.round)
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


class Seq2VecJoinVert(Seq2VecForward):
    class News:
        __slots__ = ['vertical', 'sub_vertical', 'title', 'body']

        def __init__(self, vertical, sub_vertical, title, body):
            self.vertical = vertical
            self.sub_vertical = sub_vertical
            self.title = title
            self.body = body

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

        self.docs = {int(line[1]): self.News(
            line[2],
            line[3],
            title_parser(line[4])[0],
            body_parser(line[5])[0],
        ) for line in docs}

        self.doc_count = max(self.docs.keys()) + 1
        doc_example = self.docs[self.doc_count - 1]
        self.docs[0] = self.News(
            'vertical', 'sub_vertical',
            np.zeros_like(doc_example.title),
            np.zeros_like(doc_example.body))

        self.verticals = {v: k for k, v in enumerate(set(news.vertical for news in self.docs.values()))}
        self.sub_verticals = {v: k for k, v in enumerate(set(news.vertical for news in self.docs.values()))}

        self.verticals = {k: keras.utils.to_categorical(v, len(self.verticals)) for k, v in self.verticals.items()}
        self.sub_verticals = {k: keras.utils.to_categorical(v, len(self.sub_verticals)) for k, v in
                              self.sub_verticals.items()}
        logging.info("[+] loading docs metadata")

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
                                if ch.count:
                                    clicked = ch.get_title()
                                    yield clicked, self.docs[pos].title, self.verticals[self.docs[pos].vertical], 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield clicked, self.docs[neg].title, self.verticals[self.docs[neg].vertical], 0
                                ch.push(pos)

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
                                    clicked = ch.get_title()
                                    yield clicked, self.docs[pos].title, self.verticals[self.docs[pos].vertical], 1
                                    for neg in impression.negative_samples(self.config.negative_samples):
                                        yield clicked, self.docs[neg].title, self.verticals[self.docs[neg].vertical], 0
                                ch.push(pos)

    def test_gen(self):
        def __gen__(_clicked, _impression):
            for p in _impression.pos:
                doc = self.docs[p]
                yield _clicked, doc.title, self.verticals[doc.vertical], 1
            for n in _impression.neg:
                doc = self.docs[n]
                yield _clicked, doc.title, self.verticals[doc.vertical], 0

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
                        clicked = ch.get_title()
                        yield list(__gen__(clicked, impression))
                        for pos in impression.pos:
                            ch.push(pos)

    @property
    def test(self):
        for b in self.test_gen():
            batch = [np.stack(x) for x in zip(*b)]
            yield [self.model.predict([batch[0], batch[1], batch[2]]).reshape(-1), batch[3]]

    def _build_model(self):
        model = super(Seq2VecJoinVert, self)._build_model()

        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        candidate = keras.Input((self.config.title_shape,))
        vert = keras.Input((len(self.verticals),))

        seq_logits = model([clicked, candidate])

        vert_logits = keras.layers.Dense(len(self.verticals), activation='softmax')(self.doc_encoder(candidate))
        loss = keras.losses.categorical_crossentropy(vert, vert_logits)

        self.model = keras.Model([clicked, candidate, vert], seq_logits)
        self.model.add_loss(loss)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )
