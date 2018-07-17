#!/usr/bin/env python3

import json
import logging
import os
import random

import numpy as np

from ddi.label_index import n_label
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


def load_dataset(dataset_path, verbose=False):
    # load json dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    if verbose:
        logging.info('LOAD DATASET: {} PAIRS'.format(len(dataset)))
    return dataset


def generate_batch(dataset, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(dataset)
    batch_list = []
    for i in range(0, len(dataset), batch_size):
        batch_list.append(group_batch(dataset[i:i+batch_size]))
    return batch_list


def group_batch(batch):
    label_list = []
    word_emb_input_list = []
    e1_pos_emb_input_list = []
    e2_pos_emb_input_list = []
    CUI_emb_input_list = []

    max_word_seq_len = 0
    max_CUI_seq_len = 0
    for pair in batch:
        if len(pair['index_list']) > max_word_seq_len:
            max_word_seq_len = len(pair['index_list'])
        if len(pair['cui']) > max_CUI_seq_len:
            max_CUI_seq_len = len(pair['cui'])

    for pair in batch:
        word_emb_input = pad_sequences([pair['index_list']],
                                       maxlen=max_word_seq_len)
        e1_pos_emb_input = pad_sequences([pair['offset1']],
                                         maxlen=max_word_seq_len)
        e2_pos_emb_input = pad_sequences([pair['offset2']],
                                         maxlen=max_word_seq_len)
        CUI_emb_input = pad_sequences([pair['cui']],
                                      maxlen=max_CUI_seq_len)
        pair_label = to_categorical(pair['label_ix'], num_classes=n_label())
        pair_label = pair_label[None, :]

        word_emb_input_list.append(word_emb_input)
        e1_pos_emb_input_list.append(e1_pos_emb_input)
        e2_pos_emb_input_list.append(e2_pos_emb_input)
        CUI_emb_input_list.append(CUI_emb_input)
        label_list.append(pair_label)

    word_emb_input_mat = np.concatenate(word_emb_input_list)
    e1_pos_emb_input_mat = np.concatenate(e1_pos_emb_input_list)
    e2_pos_emb_input_mat = np.concatenate(e2_pos_emb_input_list)
    CUI_emb_input_mat = np.concatenate(CUI_emb_input_list)
    label_mat = np.concatenate(label_list)
    input_dict = {'word_input': word_emb_input_mat,
                  'offset1_input': e1_pos_emb_input_mat,
                  'offset2_input': e2_pos_emb_input_mat,
                  'cui_input': CUI_emb_input_mat}
    return {'x': input_dict, 'y': label_mat}


if __name__ == '__main__':
    train_dataset_path = os.getenv('train_dataset_path')
    dataset = load_dataset(train_dataset_path)
    batch_list = generate_batch(dataset, batch_size=128)
    for batch in batch_list:
        x = batch['x']
        for k, v in x.items():
            print(k, v.shape)
        y = batch['y']
        print('y, ' + str(y.shape))
        print('-' * 80)
