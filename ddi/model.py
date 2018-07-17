#!/usr/bin/env python3

import os
from keras import optimizers
from keras.layers import (LSTM, Bidirectional, Input, Lambda, Dropout,
                          Embedding, Concatenate, Add)
from keras.layers import Dense
from keras import regularizers
from keras import backend as K
from ddi.label_index import n_label
from keras.models import Model

import numpy as np
from ddi.transformer import GetPosEncodingMatrix, Encoder, GetPadMask
from ddi.pooling import AttentionPooling


class Transformer:

    def __init__(self, config):

        # word emb
        word_emb_weights_path = os.getenv('emb_path')
        word_emb_weights = np.load(word_emb_weights_path)
        vocab_len, d_word_emb = word_emb_weights.shape
        self.word_emb_layer = Embedding(vocab_len, d_word_emb, name='word_emb',
                                        weights=[word_emb_weights],
                                        trainable=True)

        # offset emb
        offset_len = 20
        d_offset_emb = config['d_offset']
        self.offset_emb_layer = Embedding(offset_len, d_offset_emb,
                                          name='offset_emb', trainable=True)

        # cui emb
        cui_emb_path = os.getenv('cui_path')
        cui_emb_weights = np.load(cui_emb_path)
        cui_len, d_cui_emb = cui_emb_weights.shape
        self.cui_emb_layer = Embedding(cui_len, d_cui_emb, name='cui_emb',
                                       weights=[cui_emb_weights],
                                       trainable=False)
        self.input_dropout = Dropout(config['dropout']['emb'],
                                     name='input_dropout')

        # pos emb
        len_limit = 80
        d_model = 2 * config['d_hidden']
        self.d_model = d_model
        pos_emb_weights = [GetPosEncodingMatrix(len_limit, d_model)]
        self.pos_emb = Embedding(len_limit, d_model, trainable=False,
                                 weights=pos_emb_weights, name='pos_emb')

        # encoder
        d_inner_hid = config['d_ff']
        n_head = config['h']
        d_k = d_v = int(d_model / n_head)
        block_count = config['block_count']
        block_dropout = config['dropout']['block']
        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v,
                               block_count, block_dropout)

        # blstm
        d_hidden = config['d_hidden']
        # lstm_regularizer = regularizers.l2(config['lstm_weight_decay'])
        lstm = LSTM(d_hidden, return_sequences=True, name='lstm')
        self.blstm = Bidirectional(lstm, merge_mode='concat', name='blstm')
        self.blstm_dropout = Dropout(config['dropout']['lstm'])

        # encoder dropout
        self.encoder_dropout = Dropout(config['dropout']['encoder'])

        # fc
        nlabel = n_label()
        fc_regularizer = regularizers.l2(config['fc_weight_decay'])
        self.fc = Dense(nlabel, activation='softmax', name='fc',
                        kernel_regularizer=fc_regularizer,
                        bias_regularizer=fc_regularizer)

        # optimizer
        self.optimizer = optimizers.Adam(config['lr'])
        # self.optimizer = optimizers.Adam(config['lr'])

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def get_pretrained_weights(self, weights_path):
        weights = np.load(weights_path)
        weight_list = [weights[i] for i in range(len(weights))]
        return weight_list

    def compile(self):
        # inputs
        word_input = Input(shape=(None,), dtype='int32', name='word_input')
        offset1_input = Input(shape=(None,), dtype='int32',
                              name='offset1_input')
        offset2_input = Input(shape=(None,), dtype='int32',
                              name='offset2_input')
        cui_input = Input(shape=(None,), dtype='int32', name='cui_input')

        # embs
        word_out = self.word_emb_layer(word_input)
        offset1_out = self.offset_emb_layer(offset1_input)
        offset2_out = self.offset_emb_layer(offset2_input)
        cui_out = self.cui_emb_layer(cui_input)
        token_emb = Concatenate()([word_out, offset1_out, offset2_out,
                                   cui_out])

        # emb dropout
        token_emb = self.input_dropout(token_emb)

        # blstm
        blstm_out = self.blstm(token_emb)
        blstm_out = self.blstm_dropout(blstm_out)

        # pos emb
        src_pos = Lambda(self.get_pos_seq)(word_input)
        pos_out = self.pos_emb(src_pos)
        enc_in = Add()([pos_out, blstm_out])

        # encoder
        mask = Lambda(lambda x: GetPadMask(x, x))(word_input)
        enc_out = self.encoder(enc_in, mask)
        enc_out = AttentionPooling()(enc_out)
        # enc_out_head = Lambda(lambda x: x[:, 0, :])(enc_out)
        # enc_out_tail = Lambda(lambda x: x[:, -1, :])(enc_out)
        # enc_out = Concatenate()([enc_out_head, enc_out_tail])
        enc_out = self.encoder_dropout(enc_out)

        # fc
        fc_out = self.fc(enc_out)

        # build model
        model_inputs = [word_input, offset1_input, offset2_input, cui_input]
        model_outputs = [fc_out]
        self.model = Model(model_inputs, model_outputs)
        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy')


if __name__ == '__main__':
    pass
