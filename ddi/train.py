#!/usr/bin/env python3
import json
import logging
import os

import progressbar
from ddi.dataloader import generate_batch, load_dataset
from ddi.eval import evaluation
from ddi.logger import Logger
from ddi.model import Transformer
from tensorboardX import SummaryWriter


def train():
    Logger('DDI-Keras')
    writer = SummaryWriter()

    train_dataset_path = os.getenv('train_dataset_path')
    train_dataset = load_dataset(train_dataset_path, verbose=True)

    # load the config
    config_path = os.getenv('config_path')

    with open(config_path) as f:
        config = json.load(f)
    logging.info('CONFIG LOADED')
    logging.info(config)

    ddi_model = Transformer(config)
    ddi_model.compile()
    # ddi_model.model.summary(print_fn=(lambda line: logging.info(line)))

    epochs = config['epochs']
    max_f1 = 0.00
    max_f1_epoch = -1

    for epoch_ix in range(config['epochs']):
        loss_in_epoch = 0.0
        batch_list = generate_batch(train_dataset,
                                    batch_size=config['batch_size'])
        progressbar_prefix = f"EPOCH {epoch_ix+1}/{config['epochs']} "
        with progressbar.ProgressBar(max_value=len(batch_list),
                                     redirect_stdout=True,
                                     prefix=progressbar_prefix) as bar:
            for batch_ix, batch in enumerate(batch_list):
                batch_x = batch['x']
                batch_y = batch['y']
                loss_in_epoch += ddi_model.model.train_on_batch(batch_x,
                                                                batch_y)
                # result = ddi_model.model.predict_on_batch(batch_x)
                # print(result.shape)
                # return
                # for out in result:
                #     print(out.shape)
                bar.update(batch_ix)
        logging.info(f'LOSS {epoch_ix+1}/{epochs}: {loss_in_epoch}')
        writer.add_scalar('loss', loss_in_epoch, epoch_ix)

        eval_f1 = evaluation(ddi_model.model, epoch_ix, writer)
        if eval_f1 > max_f1:
            max_f1 = eval_f1
            max_f1_epoch = epoch_ix
        logging.info(f'MAX F1 (EPOCH {max_f1_epoch+1}): {max_f1}')


if __name__ == '__main__':
    train()
