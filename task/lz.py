from .seq2vec import *


class Seq2Vec2(Seq2VecForward):
    def _build_model(self):
        self.doc_encoder = doc_encoder = self.get_doc_encoder()
        user_encoder = keras.layers.TimeDistributed(doc_encoder)

        clicked = keras.Input((self.config.window_size, self.config.title_shape))
        candidate = keras.Input((self.config.title_shape,))

        clicked_vec = user_encoder(clicked)
        candidate_vec = doc_encoder(candidate)

        mask = models.ComputeMasking(0)(clicked)
        clicked_vec = keras.layers.Lambda(lambda x: x[0] * keras.backend.expand_dims(x[1]))([clicked_vec, mask])

        user_model = self.config.arch
        logging.info('[!] Selecting User Model: {}'.format(user_model))
        if user_model == 'satt':
            clicked_vec = keras.layers.Masking()(clicked_vec)
            clicked_vec = models.SelfAttention()(clicked_vec)
            clicked_vec = models.GlobalAveragePoolingMasked(mask)(clicked_vec)
        else:
            if user_model != 'avg':
                logging.warning('[!] arch {} not found, using average by default'.format(user_model))
            clicked_vec = models.GlobalAveragePoolingMasked(mask)(clicked_vec)

        join_vec = keras.layers.concatenate([clicked_vec, candidate_vec])
        hidden = keras.layers.Dense(self.config.hidden_dim, activation='relu')(join_vec)
        logits = keras.layers.Dense(1, activation='sigmoid')(hidden)

        self.model = keras.Model([clicked, candidate], logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=self.loss,
            metrics=[utils.auc_roc]
        )
