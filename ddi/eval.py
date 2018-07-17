#!/usr/bin/env python3

import json
import logging
import os

import numpy as np

import progressbar
from beautifultable import BeautifulTable
from ddi.dataloader import generate_batch, load_dataset
from ddi.label_index import ix2label, labels
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)


def evaluation(model, epoch, config, writer=None):
    logging.info('EVALUATING')
    test_dataset_path = os.getenv('test_dataset_path')
    test_dataset = load_dataset(test_dataset_path)
    batch_list = generate_batch(test_dataset,
                                batch_size=config['eval_batch_size'])
    y_true_list = []
    y_pred_list = []

    neg_path = os.getenv('test_neg_path')
    with open(neg_path) as f:
        neg_pair_list = json.load(f)
        y_pred_list = [0] * len(neg_pair_list)
        y_true_list = [pair['label_ix'] for pair in neg_pair_list]

    with progressbar.ProgressBar(max_value=len(batch_list),
                                 redirect_stdout=True) as bar:
        for batch_ix, batch in enumerate(batch_list):
            x = batch['x']
            y = batch['y']

            prediction = model.predict_on_batch(x)
            batch_pred_list = prediction.argmax(axis=-1).reshape(-1)
            batch_pred_list = batch_pred_list.tolist()
            batch_true_list = y.argmax(axis=-1).reshape(-1).tolist()
            y_true_list += batch_true_list
            y_pred_list += batch_pred_list
            bar.update(batch_ix)
    conf_mat = confusion_matrix(y_true_list, y_pred_list)
    conf_mat_table(conf_mat)
    detection_metric_table(y_true_list, y_pred_list, epoch)

    prec_list = precision_score(y_true_list, y_pred_list, average=None)
    recall_list = recall_score(y_true_list, y_pred_list, average=None)
    f1_list = f1_score(y_true_list, y_pred_list, average=None)

    class_f1 = class_metric_table(conf_mat, prec_list, recall_list, f1_list,
                                  epoch, writer)
    return class_f1


def conf_mat_table(conf_mat):
    '''Log out the confusin matrix.

    Args:
        conf_mat (array): The given confusion matrix.
    '''
    table = BeautifulTable()
    ddi_pred_type = ['pred_' + label for label in labels()]
    ddi_actual_type = ['actual_' + label for label in labels()]
    table.column_headers = ['ddi_type'] + ddi_pred_type

    for i in range(len(ddi_pred_type)):
        table.append_row([ddi_actual_type[i]] + list(conf_mat[i]))
    logging.info('\n' + str(table))


def detection_metric_table(true_list, pred_list, epoch):
    '''Log out detection metrics.

    Args:
        true_list (list): The actual lables.
        pred_list (list): The predicted lables.
        epoch (int): The current epoch.
    '''
    true_list_det = [0 if y == 0 else 1 for y in true_list]
    pred_list_det = [0 if y == 0 else 1 for y in pred_list]
    conf_mat = confusion_matrix(true_list_det, pred_list_det)
    tp = conf_mat[1, 1]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    total = tp + fn
    prec = precision_score(true_list_det, pred_list_det, average='binary')
    prec = np.round(prec, decimals=4)
    recall = recall_score(true_list_det, pred_list_det, average='binary')
    recall = np.round(recall, decimals=4)
    f1 = f1_score(true_list_det, pred_list_det, average='binary')
    f1 = np.round(f1, decimals=4)
    row = 'detection | tp:{} | fp:{} | fn:{} | total:{} |'.format(tp, fp, fn,
                                                                  total)
    row += ' prec:{} | recall:{}| f1:{} |'.format(prec, recall, f1)
    logging.info(row)


def class_metric_table(conf_mat, prec_list, recall_list, f1_list,
                       epoch, writer=None):
    '''Log out a metric table including every type.

    Args:
        conf_mat (array): Confusion matrix.
        prec_list (array): Precision list.
        recall_list (array): Recall list.
        f1_list (array): F1 score list.
        epoch (int): The current epoch.

    Returns:
        Return the classfication F1 score.
    '''
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    total_sum = 0
    for i in range(1, len(prec_list)):
        label = ix2label(i)
        tp = conf_mat[i, i]
        fp = np.sum(conf_mat[:, i]) - tp
        fn = np.sum(conf_mat[i]) - tp
        total = np.sum(conf_mat[i])
        prec = np.round(prec_list[i], decimals=4)
        recall = np.round(recall_list[i], decimals=4)
        f1 = np.round(f1_list[i], decimals=4)
        row = '{} | tp:{} | fp:{} | fn:{} | total:{} |'.format(label, tp, fp,
                                                               fn, total)
        row += ' prec:{} | recall:{}| f1:{} |'.format(prec, recall, f1)
        logging.info(row)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        total_sum += total
    class_prec = tp_sum / (tp_sum + fp_sum)
    class_prec = np.round(class_prec, decimals=4)
    class_recall = tp_sum / (tp_sum + fn_sum)
    class_recall = np.round(class_recall, decimals=4)
    class_f1 = 2 * class_prec * class_recall / (class_prec + class_recall)
    class_f1 = np.round(class_f1, decimals=4)
    if np.isnan(class_f1):
        class_f1 = 0.0
    row = 'class | tp:{} | fp:{} | fn:{} |'.format(tp_sum, fp_sum, fn_sum)
    row += ' total:{} | prec:{}'.format(total_sum, class_prec)
    row += ' | recall:{} | f1:{} |'.format(class_recall, class_f1)
    logging.info(row)
    m_prec = np.mean(prec_list[1:])
    m_prec = np.round(m_prec, decimals=4)
    m_recall = np.mean(recall_list[1:])
    m_recall = np.round(m_recall, decimals=4)
    m_f1 = 2 * m_prec * m_recall / (m_prec + m_recall)
    m_f1 = np.round(m_f1, decimals=4)
    if np.isnan(m_f1):
        m_f1 = 0.0
    row = 'macro | prec: {} | recall: {} | f1: {}'.format(m_prec, m_recall,
                                                          m_f1)

    logging.info(row)
    if writer is not None:
        writer.add_scalar('classification/prec',
                          class_prec, epoch)
        writer.add_scalar('classification/recall',
                          class_recall, epoch)
        writer.add_scalar('classification/f1', class_f1,
                          epoch)
    return class_f1


if __name__ == '__main__':
    pass
