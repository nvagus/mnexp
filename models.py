import keras
import keras.backend as K
import numpy as np
import tensorflow as tf


def simple_attention(target):
    attention = keras.layers.Dense(1, activation=keras.activations.tanh)(target)
    attention = keras.layers.Reshape((-1,))(attention)
    attention_weight = keras.layers.Activation(keras.activations.softmax)(attention)
    return keras.layers.Dot((1, 1))([target, attention_weight])


def simple_query_attention(target, query):
    attention = keras.layers.Dot(-1, -1)([target, query])
    attention_weight = keras.layers.Activation(keras.activations.softmax)(attention)
    return keras.layers.Dot((1, 1))([target, attention_weight])


class ComputeMasking(keras.layers.Layer):
    def __init__(self, mask_value=0., **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs, **kwargs):
        mask = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class GlobalAveragePoolingMasked(keras.layers.Layer):
    def __init__(self, mask, **kwargs):
        super(GlobalAveragePoolingMasked, self).__init__(**kwargs)
        self.mask = mask

    def call(self, inputs, **kwargs):
        return K.sum(inputs, axis=-2) / (K.sum(self.mask, axis=-1, keepdims=True) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]


class SimpleAttentionMasked(keras.layers.Layer):
    def __init__(self, mask, **kwargs):
        super(SimpleAttentionMasked, self).__init__(**kwargs)
        self.mask = mask

    def call(self, inputs, **kwargs):
        attention = keras.layers.Reshape((-1,))(keras.layers.Dense(1, activation=keras.activations.tanh)(inputs))
        attention = K.exp(attention) * self.mask
        # attention.set_shape(self.mask.get_shape())
        attention_weight = attention / (K.sum(attention, axis=-1, keepdims=True) + K.epsilon())

        return keras.layers.Dot((1, 1))([inputs, attention_weight])

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]


class QueryAttentionMasked(keras.layers.Layer):
    def __init__(self, mask, hidden_size=100, **kwargs):
        super(QueryAttentionMasked, self).__init__(**kwargs)
        self.mask = mask
        self.hidden_size = hidden_size

    def call(self, inputs, **kwargs):
        query_vec = keras.layers.Dense(self.hidden_size, use_bias=False)(inputs[1])
        query_vec = K.expand_dims(query_vec, axis=-2)
        attention = keras.activations.tanh(keras.layers.Dense(self.hidden_size)(inputs[0]) + query_vec)
        attention = keras.layers.Dense(1, use_bias=False)(attention)
        attention = K.squeeze(attention, axis=-1)
        attention = K.exp(attention) * self.mask
        attention_weight = attention / (K.sum(attention, axis=-1, keepdims=True) + K.epsilon())
        return keras.layers.Dot((1, 1))([inputs[0], attention_weight])

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-2] + input_shape[0][-1:]


def time_attention(target, reference, weight=None):
    size = target.get_shape()[-2].value

    def __layer__(x):
        attention = keras.layers.Dense(1)(reference)
        attention = keras.layers.Reshape((-1,))(attention)

        lam = tf.Variable(tf.constant(0.1, shape=()), constraint=keras.constraints.non_neg())
        t = tf.constant(np.arange(size), dtype=tf.float32, shape=(size,))
        if weight is None:
            attention = attention - lam * t
        else:
            attention = attention - lam * t + tf.log(weight + 1)

        attention = tf.tanh(attention)

        attention_weight = keras.layers.Activation(keras.activations.softmax)(attention)

        return keras.layers.Dot((1, 1))([x, attention_weight])

    return keras.layers.Lambda(__layer__)(target)


def caca(input_shape, filter_shape, embedding_layer, dropout):
    input_count, input_length = input_shape
    filter_count, filter_size = filter_shape

    line_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

    line_i = keras.Input((input_length,), dtype='int32')
    line_e = keras.layers.Dropout(dropout)(embedding_layer(line_i))
    line_c = line_cnn(line_e)
    line_a = simple_attention(keras.layers.Dropout(dropout)(line_c))

    encoder = keras.layers.TimeDistributed(keras.Model(line_i, line_a))
    global_cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

    global_i = keras.Input((input_count, input_length), dtype='int32')
    global_e = encoder(global_i)
    global_c = global_cnn(global_e)
    global_a = simple_attention(keras.layers.Dropout(dropout)(global_c))
    return keras.Model(global_i, global_a)


kv_split = keras.layers.Lambda(lambda x: tf.unstack(x, axis=-2))


def ea(input_shape, embedding_layer):
    line_i = keras.Input((2, input_shape))
    line_s, line_v = kv_split(line_i)
    line_e = embedding_layer(line_s)
    line_a = simple_attention(line_e)
    return keras.Model(line_i, line_a)


def han(input_shape, embedding_layer, dropout):
    line_i = keras.Input(input_shape[1:])
    line_e = keras.layers.Dropout(dropout)(embedding_layer(line_i))
    line_a = keras.layers.Dropout(dropout)(simple_attention(line_e))

    encoder = keras.layers.TimeDistributed(keras.Model(line_i, line_a))
    doc_i = keras.Input(input_shape)
    doc_e = encoder(doc_i)
    doc_a = keras.layers.Dropout(dropout)(simple_attention(doc_e))
    return keras.Model(doc_i, doc_a)


def ca(input_size, filter_shape, embedding_layer, dropout, output_dim=None, name=None):
    filter_count, filter_size = filter_shape

    i = keras.Input((input_size,), dtype='int32')
    cnn = keras.layers.Conv1D(filter_count, filter_size, padding='same', activation='relu', strides=1)

    e = keras.layers.Dropout(dropout)(embedding_layer(i))
    c = cnn(e)
    a = simple_attention(keras.layers.Dropout(dropout)(c))
    if output_dim is not None:
        a = keras.layers.Dense(output_dim)(a)

    return keras.Model(i, a, name=name)


def la(input_size, hidden_dim, embedding_layer, dropout, output_dim=None):
    i = keras.Input((input_size,), dtype='int32')
    lstm = keras.layers.LSTM(hidden_dim,
                             recurrent_dropout=dropout,
                             dropout=dropout,
                             activation='relu',
                             return_sequences=True)

    e = keras.layers.Dropout(dropout)(embedding_layer(i))
    c = lstm(e)
    a = simple_attention(keras.layers.Dropout(dropout)(c))
    if output_dim is not None:
        a = keras.layers.Dense(output_dim)(a)

    return keras.Model(i, a)


def just_dot(x1, x2, *_):
    dot = keras.layers.Dot(-1, -1)([x1, x2])
    dot = keras.layers.Activation(keras.activations.sigmoid)(dot)
    return dot, x1, x2


def layer_dot(x1, x2, dim):
    x1_ = keras.layers.Dense(dim)(x1)
    x2_ = keras.layers.Dense(dim)(x2)
    dot = keras.layers.Dot(-1, -1)([x1_, x2_])
    dot = keras.layers.Activation(keras.activations.sigmoid)(dot)
    return dot, x1_, x2_


def drop_dot(x1, x2, dropout):
    x1_ = keras.layers.Dropout(dropout)(x1)
    x2_ = keras.layers.Dropout(dropout)(x2)
    dot = keras.layers.Dot(-1, -1)([x1_, x2_])
    dot = keras.layers.Activation(keras.activations.sigmoid)(dot)
    return dot, x1_, x2_


def layer_l2(x1, x2, dim):
    x1_ = keras.layers.Dense(dim)(x1)
    x2_ = keras.layers.Dense(dim)(x2)
    dot = keras.layers.Lambda(lambda x: 1 - K.exp(-K.sum((x[0] - x[1]) ** 2, axis=-1, keepdims=True)))([x1_, x2_])
    return dot, x1_, x2_


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(keras.layers.Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.u_regularizer = keras.regularizers.get(u_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.u_constraint = keras.constraints.get(u_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class QueryAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(QueryAttention, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2

        super(QueryAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        ait = keras.layers.Dot(-1, -1)(x)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask[0], K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x[0] * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][-1]


class SelfAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(SelfAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        ait = keras.layers.Dot(-1, -1)([x, x])

        a = K.exp(ait)

        if mask is not None:
            a *= K.expand_dims(K.cast(mask, K.floatx()), axis=-2)

        a /= K.cast(K.sum(a, axis=-2, keepdims=True) + K.epsilon(), K.floatx())

        return keras.layers.dot([a, x], axes=-2)

    def compute_output_shape(self, input_shape):
        return input_shape


class SimpleQueryAttentionMasked(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True

        super(SimpleQueryAttentionMasked, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SimpleQueryAttentionMasked, self).build(input_shape)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None, **kwargs):
        x = inputs[0]
        query_vec = inputs[1]
        ait = K.batch_dot(x, query_vec)

        a = K.exp(ait)

        if mask[0] is not None:
            a *= K.cast(mask[0], K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class TanhQueryAttentionMasked(keras.layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        self.supports_masking = True
        self.hidden_size = hidden_size
        super(TanhQueryAttentionMasked, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TanhQueryAttentionMasked, self).build(input_shape)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None, **kwargs):
        query_vec = inputs[1]
        attention = keras.layers.Dense(self.hidden_size)(inputs[0])
        attention = keras.activations.tanh(attention)
        #        attention = K.batch_dot(attention,query_vec)
        attention = keras.layers.Dot((2, 1))([attention, query_vec])
        attention = K.exp(attention)
        if mask[0] is not None:
            attention *= K.cast(mask[0], K.floatx())

        attention_weight = attention / K.cast(K.sum(attention, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        #        attention_weight = attention / (K.sum(attention,axis=-1) + K.epsilon())
        return keras.layers.Dot((1, 1))([inputs[0], attention_weight])

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][-1]

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
        }
        base_config = super(TanhQueryAttentionMasked, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePoolingMaskSupport(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(GlobalAveragePoolingMaskSupport, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GlobalAveragePoolingMaskSupport, self).build(input_shape)

    def compute_mask(self, inputs, input_mask=None):
        return None

    def call(self, inputs, mask=None, **kwargs):
        if mask is not None:
            return K.sum(inputs, axis=-2) / (K.sum(K.cast(mask, K.dtype(inputs)), axis=-1, keepdims=True) + K.epsilon())
        else:
            output = K.mean(inputs, axis=-2)
            return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]


class SimpleAttentionMaskSupport(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttentionMaskSupport, self).__init__(**kwargs)
        self.supports_masking = True
        self.kernel_initializer = keras.initializers.get('glorot_uniform')
        self.bias_initializer = keras.initializers.get('zeros')
        self.activation = keras.activations.get('tanh')
        self.units = 1

    def compute_mask(self, inputs, input_mask=None):
        return None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)

        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=None,
                                    constraint=None
                                    )

        self.built = True

    def call(self, inputs, mask=None, **kwargs):
        attention = K.dot(inputs, self.kernel)
        attention = K.bias_add(attention, self.bias, data_format='channels_last')
        attention = self.activation(attention)

        attention = K.squeeze(attention, axis=2)
        if mask is not None:
            attention = K.exp(attention) * K.cast(mask, K.floatx())
        else:
            attention = K.exp(attention)

        attention_weight = attention / (K.sum(attention, axis=-1, keepdims=True) + K.epsilon())

        attention_weight = K.expand_dims(attention_weight)
        weighted_input = inputs * attention_weight
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]


class OverwriteMasking(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0] * K.expand_dims(inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class SliceAxis1(keras.layers.Layer):
    def __init__(self, index, **kwargs):
        self.supports_masking = True
        self.index = index
        super(SliceAxis1, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None, **kwargs):
        return inputs[:, self.index[0]:self.index[1]]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.index[1] - self.index[0]) + input_shape[2:]

    def get_config(self):
        config = {
            'index': self.index,
        }
        base_config = super(SliceAxis1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Split5(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.unstack(inputs, 5, axis=-2)

    def compute_output_shape(self, input_shape):
        return [input_shape[:-2] + input_shape[-1:] for _ in range(5)]


class AlphaAdd(keras.layers.Layer):
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=keras.initializers.Constant(0.5),
            constraint=keras.constraints.MinMaxNorm(0., 1.))

        super(AlphaAdd, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0] * self.alpha + inputs[1] * (1 - self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape[:-2] + input_shape[-1:]
